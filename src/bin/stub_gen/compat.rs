//! Compatibility for the syntax emitted by the stub owners, not a Python parser.
//!
//! Spynso3 owns its overloads and typing refinements. Its variadic tensor overloads
//! use `typing.Unpack` and starred tuple type arguments, whose spelling requires
//! Python 3.11. Keep that type information using the Python 3.9 backport spelling.
//! Neither pyo3-stub-gen 0.17 nor Spynso3's source renderer supplies this pass.

pub fn python_39_stub_source(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut square_brackets = Vec::new();
    let mut position = 0;
    let mut previous = None;
    let mut needs_unpack = false;
    let mut has_unpack_import = false;
    // Insert before the first ordinary statement, after comments, module
    // docstrings and __future__ imports. Generated imports are single-line.
    let mut import_position = None;
    let mut line_start = true;
    while position < bytes.len() {
        let byte = bytes[position];
        if byte == b'#' {
            let end = bytes[position..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(bytes.len(), |n| position + n);
            output.extend_from_slice(&bytes[position..end]);
            position = end;
            continue;
        }
        if byte == b'\'' || byte == b'"' {
            let end = string_end(bytes, position);
            output.extend_from_slice(&bytes[position..end]);
            position = end;
            previous = Some(byte);
            line_start = false;
            continue;
        }
        if line_start && !byte.is_ascii_whitespace() {
            let rest = &bytes[position..];
            let end = rest
                .iter()
                .position(|&b| b == b'\n' || b == b'#')
                .unwrap_or(rest.len());
            has_unpack_import |=
                rest[..end].trim_ascii_end() == b"from typing_extensions import Unpack";
            let string_prefix = [b"r\"", b"r'", b"u\"", b"u'", b"R\"", b"R'", b"U\"", b"U'"]
                .iter()
                .any(|prefix| rest.starts_with(*prefix));
            if !string_prefix && !rest.starts_with(b"from __future__ import ") {
                import_position.get_or_insert(output.len());
            }
            line_start = false;
        }
        if bytes[position..].starts_with(b"typing.Unpack")
            && (position == 0 || !identifier_byte(bytes[position - 1]))
            && bytes
                .get(position + "typing.Unpack".len())
                .is_none_or(|&b| !identifier_byte(b))
        {
            output.extend_from_slice(b"Unpack");
            position += "typing.Unpack".len();
            needs_unpack = true;
            previous = Some(b'k');
            continue;
        }
        if byte == b'*' && matches!(previous, Some(b'[' | b',' | b':')) {
            let mut start = position + 1;
            while bytes.get(start).is_some_and(u8::is_ascii_whitespace) {
                start += 1;
            }
            if let Some(tuple) = ["tuple[", "typing.Tuple[", "builtins.tuple["]
                .iter()
                .find(|tuple| bytes[start..].starts_with(tuple.as_bytes()))
            {
                output.extend_from_slice(b"Unpack[");
                output.extend_from_slice(tuple.as_bytes());
                square_brackets.push(true);
                position = start + tuple.len();
                previous = Some(b'[');
                needs_unpack = true;
                continue;
            }
        }
        if byte == b'[' {
            square_brackets.push(false);
        }
        output.push(byte);
        if byte == b']' && square_brackets.pop() == Some(true) {
            output.push(b']');
        }
        if byte == b'\n' {
            line_start = true;
        } else if !byte.is_ascii_whitespace() {
            previous = Some(byte);
        }
        position += 1;
    }
    let mut output = String::from_utf8(output).expect("the compatibility pass preserves UTF-8");
    if needs_unpack && !has_unpack_import {
        output.insert_str(
            import_position.unwrap_or(output.len()),
            "from typing_extensions import Unpack\n",
        );
    }
    output
}

fn identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'.' || !byte.is_ascii()
}

fn string_end(bytes: &[u8], start: usize) -> usize {
    let quote = bytes[start];
    let triple = bytes[start..].starts_with(&[quote; 3]);
    let width = if triple { 3 } else { 1 };
    let mut position = start + width;
    while position < bytes.len() {
        if bytes[position] == b'\\' {
            position = (position + 2).min(bytes.len());
        } else if bytes.get(position..position + width) == Some(&[quote; 3][..width]) {
            return position + width;
        } else {
            position += 1;
        }
    }
    bytes.len()
}

#[cfg(test)]
mod tests {
    use super::python_39_stub_source;

    #[test]
    fn preserves_variadic_overloads_and_nested_type_information() {
        let source = "import typing\n@typing.overload\ndef project(*xs: typing.Unpack[tuple[Concrete, *tuple[Scalar | Tensor[tuple[int, int]], ...]]]) -> Concrete: ...\n@typing.overload\ndef project(*xs: Scalar) -> Symbolic: ...\n";
        let expected = "from typing_extensions import Unpack\nimport typing\n@typing.overload\ndef project(*xs: Unpack[tuple[Concrete, Unpack[tuple[Scalar | Tensor[tuple[int, int]], ...]]]]) -> Concrete: ...\n@typing.overload\ndef project(*xs: Scalar) -> Symbolic: ...\n";
        assert_eq!(python_39_stub_source(source), expected);
        assert_eq!(python_39_stub_source(expected), expected);
    }

    #[test]
    fn leaves_documentation_literals_comments_and_ordinary_stars_untouched() {
        let source = "# typing.Unpack[*tuple[X, ...]]\nr\"\"\"An example: typing.Unpack[*tuple[X, ...]]. Café.\n\"\"\"\nimport typing\nAlias: typing.TypeAlias = \"typing.Unpack[*tuple[X, ...]]\"\ndef f(*xs: int, **kw: str) -> typing.Literal['*tuple[', 'typing.Unpack']: ...\n";
        assert_eq!(python_39_stub_source(source), source);
    }

    #[test]
    fn preserves_header_and_future_imports() {
        let source = "# generated\nr\"\"\"Module docs.\"\"\"\nfrom __future__ import annotations\nimport typing\ndef f(*xs: typing.Unpack[tuple[int, *tuple[str, ...]]]): ...\n";
        let output = python_39_stub_source(source);
        assert!(output.starts_with("# generated\nr\"\"\"Module docs.\"\"\"\nfrom __future__ import annotations\nfrom typing_extensions import Unpack\nimport typing\n"));
    }

    #[test]
    fn handles_multiple_nested_and_multiline_unpacking() {
        let source = "import typing\nX: tuple[\n *tuple[int, *typing.Tuple[str, ...]],\n *builtins.tuple[typing.Literal[']'], ...]]\n";
        let output = python_39_stub_source(source);
        assert!(output.contains("Unpack[tuple[int, Unpack[typing.Tuple[str, ...]]]]"));
        assert!(output.contains("Unpack[builtins.tuple[typing.Literal[']'], ...]]"));
    }

    #[test]
    fn avoids_duplicate_backport_import_and_identifier_substrings() {
        let source = "from typing_extensions import Unpack\nimport typing\nX: typing.Unpack[tuple[int, *tuple[str, ...]]]\nY: not_typing.Unpack[int]\n";
        let output = python_39_stub_source(source);
        assert_eq!(
            output
                .matches("from typing_extensions import Unpack")
                .count(),
            1
        );
        assert!(output.contains("Y: not_typing.Unpack[int]"));
    }

    #[test]
    fn documentation_imports_do_not_hide_the_required_backport() {
        let source = "\"\"\"Example:\nfrom typing_extensions import Unpack\n\"\"\"\nα: typing.Unpack[tuple[int, *tuple[str, ...]]]\n";
        let output = python_39_stub_source(source);
        assert_eq!(
            output
                .matches("from typing_extensions import Unpack")
                .count(),
            2
        );
        assert!(output.contains("α: Unpack[tuple[int, Unpack[tuple[str, ...]]]]"));
    }
}

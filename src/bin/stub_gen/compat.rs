//! Keep PEP 646 tuple-unpacking annotations readable by the supported Python 3.9.

pub(super) fn python_39_stub_source(source: &str) -> String {
    let mut source = source.replace("typing.Unpack[", "Unpack[");
    while let Some(start) = source.rfind("*tuple[") {
        let open = start + "*tuple".len();
        let mut depth = 0usize;
        let mut end = None;
        for (offset, ch) in source[open..].char_indices() {
            match ch {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(open + offset + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(end) = end else {
            break;
        };
        let replacement = format!("Unpack[{}]", &source[start + 1..end]);
        source.replace_range(start..end, &replacement);
    }
    if source.contains("Unpack[") && !source.contains("from typing_extensions import Unpack") {
        source = format!("from typing_extensions import Unpack\n{source}");
    }
    source
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_tuple_unpacking_keeps_the_complete_overload_shape() {
        let source = "def f(*x: typing.Unpack[tuple[A, *tuple[B | C[D], ...]]]) -> E: ...\n";
        assert_eq!(
            python_39_stub_source(source),
            "from typing_extensions import Unpack\ndef f(*x: Unpack[tuple[A, Unpack[tuple[B | C[D], ...]]]]) -> E: ...\n"
        );
    }

    #[test]
    fn ordinary_stubs_and_repeated_conversion_are_stable() {
        let source = "def f(*x: int) -> str: ...\n";
        assert_eq!(python_39_stub_source(source), source);
        let converted = python_39_stub_source("def f(*x: *tuple[int, ...]): ...\n");
        assert_eq!(python_39_stub_source(&converted), converted);
    }
}

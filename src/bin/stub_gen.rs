use pyo3_stub_gen::Result;
use symbolica_community::stub_info;

fn main() -> Result<()> {
    let hep_only = match std::env::args().skip(1).collect::<Vec<_>>().as_slice() {
        [] => false,
        [arg] if arg == "--hep-only" => true,
        _ => return Err(std::io::Error::other("Usage: stub_gen [--hep-only]").into()),
    };
    let mut stub = stub_info()?;
    let feynkit = stub
        .modules
        .remove("symbolica.community.feynkit")
        .ok_or_else(|| {
            std::io::Error::other("FeynKit did not register its Python stub inventory")
        })?;
    let hep = feynkit
        .to_string()
        .replace("symbolica.community.feynkit", "symbolica.community.hep");
    if !hep_only {
        if let Some(community) = stub.modules.get_mut("symbolica.community") {
            community.submodules.remove("feynkit");
            community.submodules.insert("hep".to_owned());
        }
        stub.generate()?;
    }
    let directory = stub.python_root.join("symbolica/community/hep");
    std::fs::create_dir_all(&directory)?;
    let source = hep
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(
        directory.join("__init__.pyi"),
        source.trim_end().to_owned() + "\n",
    )?;
    Ok(())
}

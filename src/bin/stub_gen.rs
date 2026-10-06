use pyo3_stub_gen::Result;
use symbolica_community::stub_info;

#[path = "stub_gen/compat.rs"]
mod compat;

fn main() -> Result<()> {
    let only = match std::env::args().skip(1).collect::<Vec<_>>().as_slice() {
        [] => None,
        [arg] if arg == "--hepkit-only" => Some("hepkit"),
        [arg] if arg == "--fastsecdec-only" => Some("fastsecdec"),
        [arg] if arg == "--tensor-only" || arg == "--spenso-only" => Some("tensor"),
        [arg] if arg == "--oneloop-only" => Some("oneloop"),
        [arg] if arg == "--vakint-only" => Some("vakint"),
        _ => {
            return Err(std::io::Error::other(
                "Usage: stub_gen [--hepkit-only | --fastsecdec-only | --tensor-only | --spenso-only | --oneloop-only | --vakint-only]",
            )
            .into());
        }
    };
    if only == Some("fastsecdec") && !cfg!(feature = "experimental-fastsecdec") {
        return Err(std::io::Error::other(
            "--fastsecdec-only requires --features experimental-fastsecdec",
        )
        .into());
    }
    let mut stub = stub_info()?;
    if let Some(module) = stub.modules.get_mut("symbolica.community.tensor") {
        spynso3::SpensoModule::prepare_stub_module(module);
    }
    let write_package = |module_name: &str, source: &str| -> std::io::Result<()> {
        let directory = stub.python_root.join(module_name.replace('.', "/"));
        std::fs::create_dir_all(&directory)?;
        let source = compat::python_39_stub_source(source);
        let source = source
            .lines()
            .map(str::trim_end)
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(
            directory.join("__init__.pyi"),
            source.trim_end().to_owned() + "\n",
        )
    };
    if matches!(only, Some("tensor" | "vakint")) {
        let module_name = if only == Some("tensor") {
            "symbolica.community.tensor"
        } else {
            "symbolica.community.hepkit.vakint"
        };
        let module = stub.modules.get(module_name).ok_or_else(|| {
            std::io::Error::other(format!(
                "{module_name} did not register its Python stub inventory"
            ))
        })?;
        let source = if only == Some("tensor") {
            spynso3::SpensoModule::stub_source(module)
        } else {
            module.to_string()
        };
        write_package(module_name, &source)?;
        return Ok(());
    }
    #[cfg(feature = "experimental-fastsecdec")]
    if let Some(module) = stub.modules.remove("symbolica.community.hepkit.fastsecdec") {
        let source = fastsecdec_python::stub_source(&module);
        write_package("symbolica.community.hepkit.fastsecdec", &source)?;
    }
    if only == Some("fastsecdec") {
        return Ok(());
    }
    let mut reducer = stub
        .modules
        .remove("symbolica.community.hepkit.oneloop")
        .ok_or_else(|| {
            std::io::Error::other("One-loop reducer did not register its Python stub inventory")
        })?;
    // PyO3 synthesizes equality, so it has no Rust method docstring to collect.
    for class in reducer.class.values_mut() {
        if class.name == "MasterIntegral"
            && let Some(methods) = class.methods.get_mut("__eq__")
        {
            for method in methods {
                method.doc = r#"Compare topology and symbolic kinematic arguments.

Examples
--------
>>> from symbolica import S, E
>>> from symbolica.community import hepkit as hep
>>> from symbolica.community.hepkit import oneloop
>>> d, k, p, s = S("d", "k", "p", "s")
>>> kin = hep.Kinematics(d, momenta=[k, p]).with_scalar_product(p, p, s)
>>> family = hep.IntegralFamily([k], [p], [kin.scalar_product(k, k),
...     kin.scalar_product(k-p, k-p)], kinematics=kin)
>>> reduction = oneloop.reduce(family, [1, 1])
>>> coefficient, master = reduction.terms[0]
>>> assert master == reduction.terms[0][1]

Parameters
----------
other : object
    Object to compare with this master. Equality compares the master
    topology and its symbolic kinematic arguments."#;
            }
        }
    }
    let oneloop = format!("{}\n{}", include_str!("../../stubs/oneloop.pyi"), reducer);
    write_package("symbolica.community.hepkit.oneloop", &oneloop)?;
    if only == Some("oneloop") {
        return Ok(());
    }
    let feynkit = feynkit_py::stub_info()?
        .modules
        .remove("symbolica.community.hepkit")
        .ok_or_else(|| {
            std::io::Error::other("FeynKit did not register its Python stub inventory")
        })?;
    stub.modules.remove("symbolica.community.hepkit");
    let hepkit = feynkit.to_string();
    if only != Some("hepkit") {
        if let Some(community) = stub.modules.get_mut("symbolica.community") {
            community.submodules.remove("feynkit");
            community.submodules.remove("spenso");
            community.submodules.remove("vakint");
            community.submodules.insert("tensor".to_owned());
            community.submodules.remove("hep");
        }
        // Keep the shipped canonical Symbolica stub and its community citation API.
        // Binding metadata does not include all of the canonical overloads.
        stub.modules.remove("symbolica.core");
        // These leaf modules are Python packages, so their stubs belong in __init__.pyi.
        let packages = [
            "symbolica.community.tensor",
            "symbolica.community.hepkit.vakint",
        ]
        .into_iter()
        .filter_map(|name| stub.modules.remove(name).map(|module| (name, module)))
        .collect::<Vec<_>>();
        stub.generate()?;
        for (name, module) in packages {
            let source = if name == "symbolica.community.tensor" {
                spynso3::SpensoModule::stub_source(&module)
            } else {
                module.to_string()
            };
            write_package(name, &source)?;
        }
    }
    let source = hepkit
        + "\n"
        + include_str!("../../stubs/ibp.pyi")
        + "\nfrom . import ibp as ibp\nfrom . import oneloop as oneloop\nfrom . import fastsecdec as fastsecdec\nfrom . import vakint as vakint\n";
    write_package("symbolica.community.hepkit", &source)?;
    Ok(())
}

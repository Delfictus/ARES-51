use std::env;
use std::path::PathBuf;

fn main() {
    // Generate C header file
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let header_path = PathBuf::from(&crate_dir).join("include");

    std::fs::create_dir_all(&header_path).unwrap();

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_language(cbindgen::Language::C)
        .with_pragma_once(true)
        .with_include("stdint.h")
        .with_include("stdbool.h")
        .generate()
        .expect("Unable to generate C bindings")
        .write_to_file(header_path.join("ares_csf.h"));

    // Generate gRPC code if feature is enabled and protoc is available
    #[cfg(feature = "grpc")]
    {
        // Check if protoc is available
        match std::process::Command::new("protoc")
            .arg("--version")
            .output()
        {
            Ok(_) => {
                // protoc is available, proceed with gRPC generation
                tonic_build::configure()
                    .build_server(true)
                    .build_client(true)
                    .compile(&["proto/csf.proto"], &["proto"])
                    .expect("Failed to compile gRPC definitions");
                println!("cargo:rustc-cfg=grpc_generated");
            }
            Err(_) => {
                // protoc is not available, skip gRPC generation but continue build
                println!("cargo:warning=protoc not found, skipping gRPC code generation");
                println!("cargo:warning=To enable gRPC support, install protoc: apt-get install protobuf-compiler");
            }
        }
    }
}

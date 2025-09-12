use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Python detection and configuration
    println!("cargo:rerun-if-changed=build.rs");
    
    // Check for Python installation
    let python_config = detect_python();
    
    // Locate Brian2 package
    let brian2_path = locate_brian2(&python_config);
    
    // Set up linking configuration
    configure_linking(&python_config, &brian2_path);
    
    // Configure PyO3
    pyo3_build_config::add_extension_module_link_args();
}

fn detect_python() -> PythonConfig {
    println!("cargo:warning=Detecting Python installation...");
    
    // Try multiple Python commands
    let python_cmds = vec!["python3", "python", "python3.11", "python3.10", "python3.9"];
    
    for cmd in python_cmds {
        if let Ok(output) = Command::new(cmd).arg("--version").output() {
            if output.status.success() {
                let version = String::from_utf8_lossy(&output.stdout);
                println!("cargo:warning=Found Python: {} - {}", cmd, version.trim());
                
                // Get Python config
                let config_output = Command::new(cmd)
                    .arg("-c")
                    .arg("import sys; print(sys.prefix)")
                    .output()
                    .expect("Failed to get Python prefix");
                
                let prefix = String::from_utf8_lossy(&config_output.stdout).trim().to_string();
                
                // Get include path
                let include_output = Command::new(cmd)
                    .arg("-c")
                    .arg("import sysconfig; print(sysconfig.get_path('include'))")
                    .output()
                    .expect("Failed to get Python include path");
                
                let include_path = String::from_utf8_lossy(&include_output.stdout).trim().to_string();
                
                return PythonConfig {
                    executable: cmd.to_string(),
                    prefix,
                    include_path,
                    version: version.trim().to_string(),
                };
            }
        }
    }
    
    panic!("Python installation not found! Please install Python 3.9 or later.");
}

fn locate_brian2(python: &PythonConfig) -> Option<String> {
    println!("cargo:warning=Locating Brian2 package...");
    
    // Try to import Brian2
    let output = Command::new(&python.executable)
        .arg("-c")
        .arg("import brian2; import os; print(os.path.dirname(brian2.__file__))")
        .output();
    
    match output {
        Ok(result) if result.status.success() => {
            let path = String::from_utf8_lossy(&result.stdout).trim().to_string();
            println!("cargo:warning=Found Brian2 at: {}", path);
            Some(path)
        }
        _ => {
            println!("cargo:warning=Brian2 not found. Brian2 integration will be disabled.");
            println!("cargo:warning=To enable Brian2, install it with: pip install brian2");
            None
        }
    }
}

fn configure_linking(python: &PythonConfig, brian2: &Option<String>) {
    // Set Python include directory
    println!("cargo:rustc-link-search=native={}/lib", python.prefix);
    println!("cargo:rustc-link-search=native={}/libs", python.prefix);
    
    // Configure feature flags based on what's available
    if brian2.is_some() {
        println!("cargo:rustc-cfg=feature=\"brian2_available\"");
    }
    
    // Check for CUDA
    if check_cuda_available() {
        println!("cargo:rustc-cfg=feature=\"cuda_available\"");
    }
    
    // Platform-specific linking
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=python3");
    }
    
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=python3");
    }
    
    #[cfg(target_os = "windows")]
    {
        // Windows needs different linking
        let version = python.version
            .split(' ')
            .last()
            .unwrap_or("3.9")
            .replace(".", "");
        println!("cargo:rustc-link-lib=python{}", version);
    }
}

fn check_cuda_available() -> bool {
    // Check for CUDA installation
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        println!("cargo:warning=CUDA found at: {}", cuda_path);
        return true;
    }
    
    // Check common CUDA locations
    let cuda_paths = vec![
        "/usr/local/cuda",
        "/opt/cuda",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
    ];
    
    for path in cuda_paths {
        if PathBuf::from(path).exists() {
            println!("cargo:warning=CUDA found at: {}", path);
            return true;
        }
    }
    
    false
}

struct PythonConfig {
    executable: String,
    prefix: String,
    include_path: String,
    version: String,
}
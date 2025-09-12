//! Enterprise data intake and file processing system

use crate::{EnterpriseError, EnterpriseResult, UploadConfig};
use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Multipart, Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::Arc,
};
use tokio::{
    fs::{self, File},
    io::AsyncWriteExt,
    sync::RwLock,
};
use uuid::Uuid;

/// File upload metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadMetadata {
    pub id: Uuid,
    pub filename: String,
    pub content_type: String,
    pub size: usize,
    pub uploaded_at: chrono::DateTime<chrono::Utc>,
    pub status: UploadStatus,
    pub use_case: Option<String>,
    pub batch_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UploadStatus {
    Uploading,
    Uploaded,
    Processing,
    Converted,
    Failed { reason: String },
    Completed,
}

/// Batch processing job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchJob {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub files: Vec<Uuid>,
    pub status: BatchStatus,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub use_case: String,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchStatus {
    Created,
    Processing,
    Converting,
    AwaitingConfirmation,
    Executing,
    Completed,
    Failed { reason: String },
}

/// Data intake service
pub struct IntakeService {
    config: UploadConfig,
    uploads: Arc<RwLock<HashMap<Uuid, UploadMetadata>>>,
    batches: Arc<RwLock<HashMap<Uuid, BatchJob>>>,
    temp_dir: PathBuf,
}

impl IntakeService {
    /// Create new intake service
    pub async fn new(config: UploadConfig) -> EnterpriseResult<Self> {
        let temp_dir = PathBuf::from(&config.temp_dir);
        
        // Ensure temp directory exists
        fs::create_dir_all(&temp_dir).await.map_err(|e| {
            EnterpriseError::Configuration {
                details: format!("Failed to create temp directory: {}", e),
            }
        })?;

        Ok(Self {
            config,
            uploads: Arc::new(RwLock::new(HashMap::new())),
            batches: Arc::new(RwLock::new(HashMap::new())),
            temp_dir,
        })
    }

    /// Start the intake service
    pub async fn start(&self) -> EnterpriseResult<()> {
        tracing::info!("Starting enterprise intake service");
        
        // Initialize upload tracking
        let app = self.create_routes();
        
        // Start background cleanup task
        self.start_cleanup_task().await;
        
        Ok(())
    }

    /// Stop the intake service
    pub async fn stop(&self) -> EnterpriseResult<()> {
        tracing::info!("Stopping enterprise intake service");
        Ok(())
    }

    /// Create web routes for the intake service
    pub fn create_routes(&self) -> Router {
        let state = IntakeState {
            service: Arc::new(self.clone()),
        };

        Router::new()
            .route("/upload/single", post(upload_single_file))
            .route("/upload/batch", post(upload_batch_files))
            .route("/upload/:id/status", get(get_upload_status))
            .route("/batch/create", post(create_batch_job))
            .route("/batch/:id", get(get_batch_status))
            .route("/batch/:id/process", post(process_batch))
            .route("/formats/supported", get(get_supported_formats))
            .layer(DefaultBodyLimit::max(self.config.max_file_size))
            .with_state(state)
    }

    /// Process single file upload
    pub async fn process_upload(
        &self,
        filename: String,
        content_type: String,
        data: Bytes,
        use_case: Option<String>,
    ) -> EnterpriseResult<UploadMetadata> {
        let upload_id = Uuid::new_v4();
        
        // Validate file format
        self.validate_file_format(&filename, &content_type)?;
        
        // Validate file size
        if data.len() > self.config.max_file_size {
            return Err(EnterpriseError::FileProcessing {
                reason: format!("File size {} exceeds limit {}", data.len(), self.config.max_file_size),
            });
        }

        // Save file to temp directory
        let file_path = self.temp_dir.join(format!("{}-{}", upload_id, filename));
        let mut file = File::create(&file_path).await.map_err(|e| {
            EnterpriseError::FileProcessing {
                reason: format!("Failed to create temp file: {}", e),
            }
        })?;
        
        file.write_all(&data).await.map_err(|e| {
            EnterpriseError::FileProcessing {
                reason: format!("Failed to write file data: {}", e),
            }
        })?;

        // Create metadata
        let metadata = UploadMetadata {
            id: upload_id,
            filename,
            content_type,
            size: data.len(),
            uploaded_at: chrono::Utc::now(),
            status: UploadStatus::Uploaded,
            use_case,
            batch_id: None,
        };

        // Store metadata
        self.uploads.write().await.insert(upload_id, metadata.clone());

        tracing::info!("File uploaded successfully: {} ({})", metadata.filename, upload_id);
        
        Ok(metadata)
    }

    /// Create new batch processing job
    pub async fn create_batch(
        &self,
        name: String,
        description: Option<String>,
        use_case: String,
        parameters: HashMap<String, serde_json::Value>,
    ) -> EnterpriseResult<BatchJob> {
        let batch_id = Uuid::new_v4();
        
        let batch = BatchJob {
            id: batch_id,
            name,
            description,
            files: Vec::new(),
            status: BatchStatus::Created,
            created_at: chrono::Utc::now(),
            use_case,
            parameters,
        };

        self.batches.write().await.insert(batch_id, batch.clone());
        
        tracing::info!("Batch job created: {} ({})", batch.name, batch_id);
        
        Ok(batch)
    }

    /// Add files to batch
    pub async fn add_files_to_batch(
        &self,
        batch_id: Uuid,
        file_ids: Vec<Uuid>,
    ) -> EnterpriseResult<()> {
        let mut batches = self.batches.write().await;
        let batch = batches.get_mut(&batch_id).ok_or_else(|| {
            EnterpriseError::FileProcessing {
                reason: format!("Batch {} not found", batch_id),
            }
        })?;

        // Validate all files exist
        let uploads = self.uploads.read().await;
        for file_id in &file_ids {
            if !uploads.contains_key(file_id) {
                return Err(EnterpriseError::FileProcessing {
                    reason: format!("File {} not found", file_id),
                });
            }
        }

        batch.files.extend(file_ids);
        
        Ok(())
    }

    /// Get upload status
    pub async fn get_upload_status(&self, upload_id: Uuid) -> EnterpriseResult<UploadMetadata> {
        self.uploads
            .read()
            .await
            .get(&upload_id)
            .cloned()
            .ok_or_else(|| EnterpriseError::FileProcessing {
                reason: format!("Upload {} not found", upload_id),
            })
    }

    /// Get batch status
    pub async fn get_batch_status(&self, batch_id: Uuid) -> EnterpriseResult<BatchJob> {
        self.batches
            .read()
            .await
            .get(&batch_id)
            .cloned()
            .ok_or_else(|| EnterpriseError::FileProcessing {
                reason: format!("Batch {} not found", batch_id),
            })
    }

    /// Validate file format
    fn validate_file_format(&self, filename: &str, content_type: &str) -> EnterpriseResult<()> {
        let extension = std::path::Path::new(filename)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");
            
        if !self.config.allowed_formats.contains(&extension.to_lowercase()) {
            return Err(EnterpriseError::FileProcessing {
                reason: format!("Unsupported file format: {}", extension),
            });
        }
        
        Ok(())
    }

    /// Start background cleanup task
    async fn start_cleanup_task(&self) {
        let uploads = self.uploads.clone();
        let temp_dir = self.temp_dir.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(3600)); // 1 hour
            
            loop {
                interval.tick().await;
                
                // Clean up old uploads (older than 24 hours)
                let cutoff = chrono::Utc::now() - chrono::Duration::hours(24);
                let mut to_remove = Vec::new();
                
                {
                    let uploads_guard = uploads.read().await;
                    for (id, metadata) in uploads_guard.iter() {
                        if metadata.uploaded_at < cutoff {
                            to_remove.push(*id);
                        }
                    }
                }
                
                for id in to_remove {
                    uploads.write().await.remove(&id);
                    
                    // Remove temp file
                    if let Ok(entries) = fs::read_dir(&temp_dir).await {
                        // Clean up temp files matching this upload ID
                        // Implementation would iterate through entries
                    }
                }
                
                tracing::debug!("Cleanup completed");
            }
        });
    }
}

impl Clone for IntakeService {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            uploads: self.uploads.clone(),
            batches: self.batches.clone(),
            temp_dir: self.temp_dir.clone(),
        }
    }
}

/// Shared state for web handlers
#[derive(Clone)]
struct IntakeState {
    service: Arc<IntakeService>,
}

/// Upload single file handler
async fn upload_single_file(
    State(state): State<IntakeState>,
    mut multipart: Multipart,
) -> Result<Json<UploadMetadata>, StatusCode> {
    while let Some(field) = multipart.next_field().await.map_err(|_| StatusCode::BAD_REQUEST)? {
        if field.name() == Some("file") {
            let filename = field.file_name().unwrap_or("unknown").to_string();
            let content_type = field.content_type().unwrap_or("application/octet-stream").to_string();
            let data = field.bytes().await.map_err(|_| StatusCode::BAD_REQUEST)?;
            
            match state.service.process_upload(filename, content_type, data, None).await {
                Ok(metadata) => return Ok(Json(metadata)),
                Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
            }
        }
    }
    
    Err(StatusCode::BAD_REQUEST)
}

/// Upload batch files handler
async fn upload_batch_files(
    State(state): State<IntakeState>,
    mut multipart: Multipart,
) -> Result<Json<Vec<UploadMetadata>>, StatusCode> {
    let mut results = Vec::new();
    
    while let Some(field) = multipart.next_field().await.map_err(|_| StatusCode::BAD_REQUEST)? {
        if field.name() == Some("files") {
            let filename = field.file_name().unwrap_or("unknown").to_string();
            let content_type = field.content_type().unwrap_or("application/octet-stream").to_string();
            let data = field.bytes().await.map_err(|_| StatusCode::BAD_REQUEST)?;
            
            match state.service.process_upload(filename, content_type, data, None).await {
                Ok(metadata) => results.push(metadata),
                Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
            }
        }
    }
    
    if results.is_empty() {
        Err(StatusCode::BAD_REQUEST)
    } else {
        Ok(Json(results))
    }
}

/// Get upload status handler
async fn get_upload_status(
    State(state): State<IntakeState>,
    Path(upload_id): Path<Uuid>,
) -> Result<Json<UploadMetadata>, StatusCode> {
    match state.service.get_upload_status(upload_id).await {
        Ok(metadata) => Ok(Json(metadata)),
        Err(_) => Err(StatusCode::NOT_FOUND),
    }
}

/// Create batch job request
#[derive(Debug, Deserialize)]
struct CreateBatchRequest {
    name: String,
    description: Option<String>,
    use_case: String,
    parameters: HashMap<String, serde_json::Value>,
}

/// Create batch job handler
async fn create_batch_job(
    State(state): State<IntakeState>,
    Json(request): Json<CreateBatchRequest>,
) -> Result<Json<BatchJob>, StatusCode> {
    match state.service.create_batch(
        request.name,
        request.description,
        request.use_case,
        request.parameters,
    ).await {
        Ok(batch) => Ok(Json(batch)),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// Get batch status handler
async fn get_batch_status(
    State(state): State<IntakeState>,
    Path(batch_id): Path<Uuid>,
) -> Result<Json<BatchJob>, StatusCode> {
    match state.service.get_batch_status(batch_id).await {
        Ok(batch) => Ok(Json(batch)),
        Err(_) => Err(StatusCode::NOT_FOUND),
    }
}

/// Process batch request
#[derive(Debug, Deserialize)]
struct ProcessBatchRequest {
    file_ids: Vec<Uuid>,
}

/// Process batch handler
async fn process_batch(
    State(state): State<IntakeState>,
    Path(batch_id): Path<Uuid>,
    Json(request): Json<ProcessBatchRequest>,
) -> Result<Json<BatchJob>, StatusCode> {
    // Add files to batch
    if let Err(_) = state.service.add_files_to_batch(batch_id, request.file_ids).await {
        return Err(StatusCode::BAD_REQUEST);
    }
    
    // Return updated batch status
    match state.service.get_batch_status(batch_id).await {
        Ok(batch) => Ok(Json(batch)),
        Err(_) => Err(StatusCode::NOT_FOUND),
    }
}

/// Get supported formats
async fn get_supported_formats(
    State(state): State<IntakeState>,
) -> Json<Vec<String>> {
    Json(state.service.config.allowed_formats.clone())
}

/// Upload query parameters
#[derive(Debug, Deserialize)]
struct UploadQuery {
    use_case: Option<String>,
    batch_id: Option<Uuid>,
}

/// File processing utilities
impl IntakeService {
    /// Detect file format from content
    pub async fn detect_format(&self, file_path: &PathBuf) -> EnterpriseResult<String> {
        let content = fs::read_to_string(file_path).await.map_err(|e| {
            EnterpriseError::FileProcessing {
                reason: format!("Failed to read file: {}", e),
            }
        })?;

        // Simple format detection
        if content.trim_start().starts_with('{') || content.trim_start().starts_with('[') {
            Ok("json".to_string())
        } else if content.contains(',') && content.lines().count() > 1 {
            Ok("csv".to_string())
        } else if content.trim_start().starts_with('<') {
            Ok("xml".to_string())
        } else if content.contains("---") {
            Ok("yaml".to_string())
        } else {
            Ok("text".to_string())
        }
    }

    /// Validate file content
    pub async fn validate_content(&self, file_path: &PathBuf, format: &str) -> EnterpriseResult<bool> {
        match format {
            "json" => {
                let content = fs::read_to_string(file_path).await.map_err(|e| {
                    EnterpriseError::FileProcessing {
                        reason: format!("Failed to read JSON file: {}", e),
                    }
                })?;
                
                serde_json::from_str::<serde_json::Value>(&content).map_err(|e| {
                    EnterpriseError::FileProcessing {
                        reason: format!("Invalid JSON format: {}", e),
                    }
                })?;
                
                Ok(true)
            }
            "csv" => {
                let content = fs::read_to_string(file_path).await.map_err(|e| {
                    EnterpriseError::FileProcessing {
                        reason: format!("Failed to read CSV file: {}", e),
                    }
                })?;
                
                let mut reader = csv::Reader::from_reader(content.as_bytes());
                reader.headers().map_err(|e| {
                    EnterpriseError::FileProcessing {
                        reason: format!("Invalid CSV format: {}", e),
                    }
                })?;
                
                Ok(true)
            }
            "xml" => {
                let content = fs::read_to_string(file_path).await.map_err(|e| {
                    EnterpriseError::FileProcessing {
                        reason: format!("Failed to read XML file: {}", e),
                    }
                })?;
                
                roxmltree::Document::parse(&content).map_err(|e| {
                    EnterpriseError::FileProcessing {
                        reason: format!("Invalid XML format: {}", e),
                    }
                })?;
                
                Ok(true)
            }
            _ => Ok(true), // Accept other formats for now
        }
    }

    /// Get file statistics
    pub async fn get_file_stats(&self, file_path: &PathBuf) -> EnterpriseResult<FileStats> {
        let metadata = fs::metadata(file_path).await.map_err(|e| {
            EnterpriseError::FileProcessing {
                reason: format!("Failed to get file metadata: {}", e),
            }
        })?;

        let format = self.detect_format(file_path).await?;
        let line_count = self.count_lines(file_path).await?;

        Ok(FileStats {
            size: metadata.len(),
            format,
            line_count,
            created: metadata.created().ok(),
            modified: metadata.modified().ok(),
        })
    }

    /// Count lines in file
    async fn count_lines(&self, file_path: &PathBuf) -> EnterpriseResult<usize> {
        let content = fs::read_to_string(file_path).await.map_err(|e| {
            EnterpriseError::FileProcessing {
                reason: format!("Failed to read file for line counting: {}", e),
            }
        })?;

        Ok(content.lines().count())
    }
}

/// File statistics
#[derive(Debug, Serialize)]
pub struct FileStats {
    pub size: u64,
    pub format: String,
    pub line_count: usize,
    pub created: Option<std::time::SystemTime>,
    pub modified: Option<std::time::SystemTime>,
}
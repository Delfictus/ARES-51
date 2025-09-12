//! Data format conversion pipeline to ARES PhasePacket format

use crate::{ConversionConfig, EnterpriseError, EnterpriseResult};
use csf_protocol::{PacketFlags, PacketPayload, PhasePacket, PhasePacketHeader};
use csf_shared_types::{ComponentId, NanoTime, PacketId, PacketType};
use csf_time::hardware_timestamp;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

/// Data conversion engine
pub struct ConversionEngine {
    config: ConversionConfig,
    converters: HashMap<String, Box<dyn DataConverter + Send + Sync>>,
    schema_cache: HashMap<String, DetectedSchema>,
}

impl ConversionEngine {
    /// Create new conversion engine
    pub fn new(config: ConversionConfig) -> EnterpriseResult<Self> {
        let mut converters: HashMap<String, Box<dyn DataConverter + Send + Sync>> = HashMap::new();
        
        // Register built-in converters
        converters.insert("json".to_string(), Box::new(JsonConverter::new()));
        converters.insert("csv".to_string(), Box::new(CsvConverter::new()));
        converters.insert("xml".to_string(), Box::new(XmlConverter::new()));
        converters.insert("yaml".to_string(), Box::new(YamlConverter::new()));
        converters.insert("binary".to_string(), Box::new(BinaryConverter::new()));

        Ok(Self {
            config,
            converters,
            schema_cache: HashMap::new(),
        })
    }

    /// Convert file to ARES PhasePacket format
    pub async fn convert_file(
        &mut self,
        file_path: &PathBuf,
        format: &str,
        use_case: &str,
    ) -> EnterpriseResult<ConversionResult> {
        // Get appropriate converter
        let converter = self.converters.get(format).ok_or_else(|| {
            EnterpriseError::ConversionFailed {
                format: format.to_string(),
            }
        })?;

        // Detect or retrieve schema
        let schema = if self.config.auto_detect_schema {
            self.detect_schema(file_path, format).await?
        } else {
            self.get_cached_schema(format, use_case)?
        };

        // Perform conversion
        let conversion_context = ConversionContext {
            file_path: file_path.clone(),
            format: format.to_string(),
            use_case: use_case.to_string(),
            schema: schema.clone(),
            preserve_metadata: self.config.preserve_metadata,
        };

        let mut attempt = 0;
        while attempt < self.config.max_conversion_attempts {
            match converter.convert(&conversion_context).await {
                Ok(packets) => {
                    // Validate conversion quality
                    let quality_score = self.validate_conversion_quality(&packets, &schema).await?;
                    
                    if quality_score >= self.config.validation_threshold {
                        return Ok(ConversionResult {
                            packets,
                            schema,
                            quality_score,
                            metadata: ConversionMetadata {
                                original_format: format.to_string(),
                                use_case: use_case.to_string(),
                                conversion_time: hardware_timestamp(),
                                records_processed: packets.len(),
                                data_integrity: quality_score,
                            },
                        });
                    }
                }
                Err(e) => {
                    tracing::warn!("Conversion attempt {} failed: {}", attempt + 1, e);
                }
            }
            
            attempt += 1;
        }

        Err(EnterpriseError::ConversionFailed {
            format: format.to_string(),
        })
    }

    /// Detect data schema from file
    async fn detect_schema(&mut self, file_path: &PathBuf, format: &str) -> EnterpriseResult<DetectedSchema> {
        let cache_key = format!("{}:{}", format, file_path.display());
        
        if let Some(cached) = self.schema_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let schema = match format {
            "json" => self.detect_json_schema(file_path).await?,
            "csv" => self.detect_csv_schema(file_path).await?,
            "xml" => self.detect_xml_schema(file_path).await?,
            "yaml" => self.detect_yaml_schema(file_path).await?,
            _ => DetectedSchema::default(),
        };

        self.schema_cache.insert(cache_key, schema.clone());
        Ok(schema)
    }

    /// Detect JSON schema
    async fn detect_json_schema(&self, file_path: &PathBuf) -> EnterpriseResult<DetectedSchema> {
        let content = tokio::fs::read_to_string(file_path).await.map_err(|e| {
            EnterpriseError::FileProcessing {
                reason: format!("Failed to read JSON file: {}", e),
            }
        })?;

        let value: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
            EnterpriseError::ConversionFailed {
                format: format!("Invalid JSON: {}", e),
            }
        })?;

        let mut schema = DetectedSchema::default();
        schema.format = "json".to_string();
        
        // Analyze JSON structure
        match &value {
            serde_json::Value::Array(arr) => {
                schema.is_array = true;
                schema.record_count = arr.len();
                
                if let Some(first) = arr.first() {
                    schema.fields = self.extract_json_fields(first);
                }
            }
            serde_json::Value::Object(_) => {
                schema.is_array = false;
                schema.record_count = 1;
                schema.fields = self.extract_json_fields(&value);
            }
            _ => {
                return Err(EnterpriseError::ConversionFailed {
                    format: "JSON must be object or array".to_string(),
                });
            }
        }

        Ok(schema)
    }

    /// Extract field information from JSON value
    fn extract_json_fields(&self, value: &serde_json::Value) -> Vec<FieldInfo> {
        let mut fields = Vec::new();
        
        if let serde_json::Value::Object(obj) = value {
            for (key, val) in obj {
                fields.push(FieldInfo {
                    name: key.clone(),
                    field_type: self.json_value_to_type(val),
                    nullable: val.is_null(),
                    array: val.is_array(),
                });
            }
        }
        
        fields
    }

    /// Convert JSON value to field type
    fn json_value_to_type(&self, value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::String(_) => "string".to_string(),
            serde_json::Value::Number(n) => {
                if n.is_f64() {
                    "float".to_string()
                } else {
                    "integer".to_string()
                }
            }
            serde_json::Value::Bool(_) => "boolean".to_string(),
            serde_json::Value::Array(_) => "array".to_string(),
            serde_json::Value::Object(_) => "object".to_string(),
            serde_json::Value::Null => "null".to_string(),
        }
    }

    /// Detect CSV schema
    async fn detect_csv_schema(&self, file_path: &PathBuf) -> EnterpriseResult<DetectedSchema> {
        let content = tokio::fs::read_to_string(file_path).await.map_err(|e| {
            EnterpriseError::FileProcessing {
                reason: format!("Failed to read CSV file: {}", e),
            }
        })?;

        let mut reader = csv::Reader::from_reader(content.as_bytes());
        let headers = reader.headers().map_err(|e| {
            EnterpriseError::ConversionFailed {
                format: format!("Invalid CSV headers: {}", e),
            }
        })?;

        let mut schema = DetectedSchema::default();
        schema.format = "csv".to_string();
        schema.is_array = true;

        // Analyze field types from first few rows
        let mut records = reader.records();
        let mut sample_records = Vec::new();
        
        for _ in 0..10 {
            if let Some(Ok(record)) = records.next() {
                sample_records.push(record);
            } else {
                break;
            }
        }

        schema.record_count = sample_records.len();

        // Infer types from sample data
        for (i, header) in headers.iter().enumerate() {
            let field_type = self.infer_csv_field_type(&sample_records, i);
            
            schema.fields.push(FieldInfo {
                name: header.to_string(),
                field_type,
                nullable: false, // CSV analysis would need null detection
                array: false,
            });
        }

        Ok(schema)
    }

    /// Infer CSV field type from sample data
    fn infer_csv_field_type(&self, records: &[csv::StringRecord], column_index: usize) -> String {
        let mut is_numeric = true;
        let mut is_integer = true;
        let mut is_boolean = true;

        for record in records {
            if let Some(value) = record.get(column_index) {
                let trimmed = value.trim();
                
                if trimmed.is_empty() {
                    continue;
                }

                // Check if it's a number
                if trimmed.parse::<f64>().is_err() {
                    is_numeric = false;
                }
                
                // Check if it's an integer
                if trimmed.parse::<i64>().is_err() {
                    is_integer = false;
                }
                
                // Check if it's a boolean
                if !matches!(trimmed.to_lowercase().as_str(), "true" | "false" | "1" | "0" | "yes" | "no") {
                    is_boolean = false;
                }
            }
        }

        if is_integer {
            "integer".to_string()
        } else if is_numeric {
            "float".to_string()
        } else if is_boolean {
            "boolean".to_string()
        } else {
            "string".to_string()
        }
    }

    /// Detect XML schema (simplified)
    async fn detect_xml_schema(&self, file_path: &PathBuf) -> EnterpriseResult<DetectedSchema> {
        let content = tokio::fs::read_to_string(file_path).await.map_err(|e| {
            EnterpriseError::FileProcessing {
                reason: format!("Failed to read XML file: {}", e),
            }
        })?;

        let doc = roxmltree::Document::parse(&content).map_err(|e| {
            EnterpriseError::ConversionFailed {
                format: format!("Invalid XML: {}", e),
            }
        })?;

        let mut schema = DetectedSchema::default();
        schema.format = "xml".to_string();
        
        // Simple XML analysis - count root elements
        schema.record_count = doc.root().children().filter(|n| n.is_element()).count();
        
        // Extract field names from first element
        if let Some(first_element) = doc.root().children().find(|n| n.is_element()) {
            for child in first_element.children().filter(|n| n.is_element()) {
                schema.fields.push(FieldInfo {
                    name: child.tag_name().name().to_string(),
                    field_type: "string".to_string(), // Simplified
                    nullable: false,
                    array: false,
                });
            }
        }

        Ok(schema)
    }

    /// Detect YAML schema
    async fn detect_yaml_schema(&self, file_path: &PathBuf) -> EnterpriseResult<DetectedSchema> {
        let content = tokio::fs::read_to_string(file_path).await.map_err(|e| {
            EnterpriseError::FileProcessing {
                reason: format!("Failed to read YAML file: {}", e),
            }
        })?;

        let value: serde_yaml::Value = serde_yaml::from_str(&content).map_err(|e| {
            EnterpriseError::ConversionFailed {
                format: format!("Invalid YAML: {}", e),
            }
        })?;

        let mut schema = DetectedSchema::default();
        schema.format = "yaml".to_string();

        // Convert YAML to JSON-like analysis
        let json_value = self.yaml_to_json_value(&value);
        
        match &json_value {
            serde_json::Value::Array(arr) => {
                schema.is_array = true;
                schema.record_count = arr.len();
                
                if let Some(first) = arr.first() {
                    schema.fields = self.extract_json_fields(first);
                }
            }
            serde_json::Value::Object(_) => {
                schema.is_array = false;
                schema.record_count = 1;
                schema.fields = self.extract_json_fields(&json_value);
            }
            _ => {}
        }

        Ok(schema)
    }

    /// Convert YAML value to JSON value for analysis
    fn yaml_to_json_value(&self, yaml_value: &serde_yaml::Value) -> serde_json::Value {
        match yaml_value {
            serde_yaml::Value::Null => serde_json::Value::Null,
            serde_yaml::Value::Bool(b) => serde_json::Value::Bool(*b),
            serde_yaml::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    serde_json::Value::Number(serde_json::Number::from(i))
                } else if let Some(f) = n.as_f64() {
                    serde_json::Number::from_f64(f)
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::Null)
                } else {
                    serde_json::Value::Null
                }
            }
            serde_yaml::Value::String(s) => serde_json::Value::String(s.clone()),
            serde_yaml::Value::Sequence(seq) => {
                let arr: Vec<serde_json::Value> = seq.iter()
                    .map(|v| self.yaml_to_json_value(v))
                    .collect();
                serde_json::Value::Array(arr)
            }
            serde_yaml::Value::Mapping(map) => {
                let mut obj = serde_json::Map::new();
                for (k, v) in map {
                    if let serde_yaml::Value::String(key) = k {
                        obj.insert(key.clone(), self.yaml_to_json_value(v));
                    }
                }
                serde_json::Value::Object(obj)
            }
            serde_yaml::Value::Tagged(tagged) => self.yaml_to_json_value(&tagged.value),
        }
    }

    /// Get cached schema for format and use case
    fn get_cached_schema(&self, format: &str, use_case: &str) -> EnterpriseResult<DetectedSchema> {
        let cache_key = format!("{}:{}", format, use_case);
        
        self.schema_cache.get(&cache_key).cloned().ok_or_else(|| {
            EnterpriseError::ConversionFailed {
                format: format!("No cached schema for {} ({})", format, use_case),
            }
        })
    }

    /// Validate conversion quality
    async fn validate_conversion_quality(
        &self,
        packets: &[PhasePacket],
        schema: &DetectedSchema,
    ) -> EnterpriseResult<f64> {
        if packets.is_empty() {
            return Ok(0.0);
        }

        let mut quality_score = 1.0;

        // Check record count consistency
        let expected_records = schema.record_count;
        let actual_records = packets.len();
        
        if expected_records > 0 {
            let record_ratio = actual_records as f64 / expected_records as f64;
            quality_score *= record_ratio.min(1.0);
        }

        // Check field coverage
        let expected_fields = schema.fields.len();
        if expected_fields > 0 {
            let mut field_coverage = 0.0;
            
            for packet in packets.iter().take(10) { // Sample first 10 packets
                let metadata_fields = packet.payload.metadata.len();
                field_coverage += (metadata_fields as f64 / expected_fields as f64).min(1.0);
            }
            
            field_coverage /= packets.len().min(10) as f64;
            quality_score *= field_coverage;
        }

        Ok(quality_score.clamp(0.0, 1.0))
    }
}

/// Data converter trait
#[async_trait::async_trait]
pub trait DataConverter {
    async fn convert(&self, context: &ConversionContext) -> EnterpriseResult<Vec<PhasePacket>>;
    fn supports_format(&self, format: &str) -> bool;
    fn get_format_name(&self) -> &str;
}

/// Conversion context
#[derive(Debug, Clone)]
pub struct ConversionContext {
    pub file_path: PathBuf,
    pub format: String,
    pub use_case: String,
    pub schema: DetectedSchema,
    pub preserve_metadata: bool,
}

/// Conversion result
#[derive(Debug)]
pub struct ConversionResult {
    pub packets: Vec<PhasePacket>,
    pub schema: DetectedSchema,
    pub quality_score: f64,
    pub metadata: ConversionMetadata,
}

/// Conversion metadata
#[derive(Debug, Serialize, Deserialize)]
pub struct ConversionMetadata {
    pub original_format: String,
    pub use_case: String,
    pub conversion_time: NanoTime,
    pub records_processed: usize,
    pub data_integrity: f64,
}

/// Detected data schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedSchema {
    pub format: String,
    pub is_array: bool,
    pub record_count: usize,
    pub fields: Vec<FieldInfo>,
    pub metadata: HashMap<String, String>,
}

impl Default for DetectedSchema {
    fn default() -> Self {
        Self {
            format: "unknown".to_string(),
            is_array: false,
            record_count: 0,
            fields: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Field information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldInfo {
    pub name: String,
    pub field_type: String,
    pub nullable: bool,
    pub array: bool,
}

/// JSON converter implementation
pub struct JsonConverter;

impl JsonConverter {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl DataConverter for JsonConverter {
    async fn convert(&self, context: &ConversionContext) -> EnterpriseResult<Vec<PhasePacket>> {
        let content = tokio::fs::read_to_string(&context.file_path).await.map_err(|e| {
            EnterpriseError::FileProcessing {
                reason: format!("Failed to read JSON file: {}", e),
            }
        })?;

        let value: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
            EnterpriseError::ConversionFailed {
                format: format!("Invalid JSON: {}", e),
            }
        })?;

        let mut packets = Vec::new();

        match value {
            serde_json::Value::Array(arr) => {
                for (index, item) in arr.into_iter().enumerate() {
                    let packet = self.json_value_to_packet(item, index, &context.use_case)?;
                    packets.push(packet);
                }
            }
            serde_json::Value::Object(_) => {
                let packet = self.json_value_to_packet(value, 0, &context.use_case)?;
                packets.push(packet);
            }
            _ => {
                return Err(EnterpriseError::ConversionFailed {
                    format: "JSON must be object or array".to_string(),
                });
            }
        }

        Ok(packets)
    }

    fn supports_format(&self, format: &str) -> bool {
        format == "json"
    }

    fn get_format_name(&self) -> &str {
        "json"
    }
}

impl JsonConverter {
    /// Convert JSON value to PhasePacket
    fn json_value_to_packet(
        &self,
        value: serde_json::Value,
        sequence: usize,
        use_case: &str,
    ) -> EnterpriseResult<PhasePacket> {
        let packet_id = PacketId::new_v4();
        let component_id = ComponentId::new_v4();
        let timestamp = hardware_timestamp();

        let mut metadata = HashMap::new();
        metadata.insert("use_case".to_string(), serde_json::Value::String(use_case.to_string()));
        metadata.insert("original_format".to_string(), serde_json::Value::String("json".to_string()));
        metadata.insert("conversion_timestamp".to_string(), serde_json::Value::String(timestamp.to_string()));

        // Extract data from JSON
        if let serde_json::Value::Object(obj) = &value {
            for (key, val) in obj {
                metadata.insert(key.clone(), val.clone());
            }
        }

        let data = serde_json::to_vec(&value).map_err(|e| {
            EnterpriseError::ConversionFailed {
                format: format!("Failed to serialize JSON data: {}", e),
            }
        })?;

        Ok(PhasePacket {
            header: PhasePacketHeader {
                packet_id,
                sequence: sequence as u64,
                timestamp,
                source_id: component_id,
                packet_type: PacketType::Data,
                flags: PacketFlags::empty(),
                priority: 5,
            },
            payload: PacketPayload {
                data,
                metadata,
            },
        })
    }
}

/// CSV converter implementation
pub struct CsvConverter;

impl CsvConverter {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl DataConverter for CsvConverter {
    async fn convert(&self, context: &ConversionContext) -> EnterpriseResult<Vec<PhasePacket>> {
        let content = tokio::fs::read_to_string(&context.file_path).await.map_err(|e| {
            EnterpriseError::FileProcessing {
                reason: format!("Failed to read CSV file: {}", e),
            }
        })?;

        let mut reader = csv::Reader::from_reader(content.as_bytes());
        let headers = reader.headers().map_err(|e| {
            EnterpriseError::ConversionFailed {
                format: format!("Invalid CSV headers: {}", e),
            }
        })?.clone();

        let mut packets = Vec::new();

        for (index, result) in reader.records().enumerate() {
            let record = result.map_err(|e| {
                EnterpriseError::ConversionFailed {
                    format: format!("Invalid CSV record: {}", e),
                }
            })?;

            let packet = self.csv_record_to_packet(&headers, &record, index, &context.use_case)?;
            packets.push(packet);
        }

        Ok(packets)
    }

    fn supports_format(&self, format: &str) -> bool {
        format == "csv"
    }

    fn get_format_name(&self) -> &str {
        "csv"
    }
}

impl CsvConverter {
    /// Convert CSV record to PhasePacket
    fn csv_record_to_packet(
        &self,
        headers: &csv::StringRecord,
        record: &csv::StringRecord,
        sequence: usize,
        use_case: &str,
    ) -> EnterpriseResult<PhasePacket> {
        let packet_id = PacketId::new_v4();
        let component_id = ComponentId::new_v4();
        let timestamp = hardware_timestamp();

        let mut metadata = HashMap::new();
        metadata.insert("use_case".to_string(), serde_json::Value::String(use_case.to_string()));
        metadata.insert("original_format".to_string(), serde_json::Value::String("csv".to_string()));
        metadata.insert("conversion_timestamp".to_string(), serde_json::Value::String(timestamp.to_string()));

        // Map CSV fields to metadata
        for (header, value) in headers.iter().zip(record.iter()) {
            metadata.insert(header.to_string(), serde_json::Value::String(value.to_string()));
        }

        // Serialize record as JSON for data payload
        let record_map: HashMap<String, String> = headers.iter()
            .zip(record.iter())
            .map(|(h, v)| (h.to_string(), v.to_string()))
            .collect();

        let data = serde_json::to_vec(&record_map).map_err(|e| {
            EnterpriseError::ConversionFailed {
                format: format!("Failed to serialize CSV record: {}", e),
            }
        })?;

        Ok(PhasePacket {
            header: PhasePacketHeader {
                packet_id,
                sequence: sequence as u64,
                timestamp,
                source_id: component_id,
                packet_type: PacketType::Data,
                flags: PacketFlags::empty(),
                priority: 5,
            },
            payload: PacketPayload {
                data,
                metadata,
            },
        })
    }
}

/// XML converter implementation
pub struct XmlConverter;

impl XmlConverter {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl DataConverter for XmlConverter {
    async fn convert(&self, context: &ConversionContext) -> EnterpriseResult<Vec<PhasePacket>> {
        let content = tokio::fs::read_to_string(&context.file_path).await.map_err(|e| {
            EnterpriseError::FileProcessing {
                reason: format!("Failed to read XML file: {}", e),
            }
        })?;

        let doc = roxmltree::Document::parse(&content).map_err(|e| {
            EnterpriseError::ConversionFailed {
                format: format!("Invalid XML: {}", e),
            }
        })?;

        let mut packets = Vec::new();

        for (index, node) in doc.root().children().filter(|n| n.is_element()).enumerate() {
            let packet = self.xml_node_to_packet(&node, index, &context.use_case)?;
            packets.push(packet);
        }

        Ok(packets)
    }

    fn supports_format(&self, format: &str) -> bool {
        format == "xml"
    }

    fn get_format_name(&self) -> &str {
        "xml"
    }
}

impl XmlConverter {
    /// Convert XML node to PhasePacket
    fn xml_node_to_packet(
        &self,
        node: &roxmltree::Node,
        sequence: usize,
        use_case: &str,
    ) -> EnterpriseResult<PhasePacket> {
        let packet_id = PacketId::new_v4();
        let component_id = ComponentId::new_v4();
        let timestamp = hardware_timestamp();

        let mut metadata = HashMap::new();
        metadata.insert("use_case".to_string(), serde_json::Value::String(use_case.to_string()));
        metadata.insert("original_format".to_string(), serde_json::Value::String("xml".to_string()));
        metadata.insert("conversion_timestamp".to_string(), serde_json::Value::String(timestamp.to_string()));
        metadata.insert("element_name".to_string(), serde_json::Value::String(node.tag_name().name().to_string()));

        // Extract attributes
        for attr in node.attributes() {
            metadata.insert(
                format!("attr_{}", attr.name()),
                serde_json::Value::String(attr.value().to_string()),
            );
        }

        // Extract child elements
        for child in node.children().filter(|n| n.is_element()) {
            if let Some(text) = child.text() {
                metadata.insert(
                    child.tag_name().name().to_string(),
                    serde_json::Value::String(text.to_string()),
                );
            }
        }

        // Use original XML as data payload
        let data = content.as_bytes().to_vec();

        Ok(PhasePacket {
            header: PhasePacketHeader {
                packet_id,
                sequence: sequence as u64,
                timestamp,
                source_id: component_id,
                packet_type: PacketType::Data,
                flags: PacketFlags::empty(),
                priority: 5,
            },
            payload: PacketPayload {
                data,
                metadata,
            },
        })
    }
}

/// YAML converter implementation
pub struct YamlConverter;

impl YamlConverter {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl DataConverter for YamlConverter {
    async fn convert(&self, context: &ConversionContext) -> EnterpriseResult<Vec<PhasePacket>> {
        // Similar to JSON converter but for YAML
        let content = tokio::fs::read_to_string(&context.file_path).await.map_err(|e| {
            EnterpriseError::FileProcessing {
                reason: format!("Failed to read YAML file: {}", e),
            }
        })?;

        let value: serde_yaml::Value = serde_yaml::from_str(&content).map_err(|e| {
            EnterpriseError::ConversionFailed {
                format: format!("Invalid YAML: {}", e),
            }
        })?;

        let mut packets = Vec::new();

        // Convert YAML to JSON for processing
        let json_content = serde_json::to_string(&value).map_err(|e| {
            EnterpriseError::ConversionFailed {
                format: format!("YAML to JSON conversion failed: {}", e),
            }
        })?;

        let json_value: serde_json::Value = serde_json::from_str(&json_content).unwrap();

        match json_value {
            serde_json::Value::Array(arr) => {
                for (index, item) in arr.into_iter().enumerate() {
                    let packet = self.yaml_value_to_packet(item, index, &context.use_case)?;
                    packets.push(packet);
                }
            }
            _ => {
                let packet = self.yaml_value_to_packet(json_value, 0, &context.use_case)?;
                packets.push(packet);
            }
        }

        Ok(packets)
    }

    fn supports_format(&self, format: &str) -> bool {
        format == "yaml" || format == "yml"
    }

    fn get_format_name(&self) -> &str {
        "yaml"
    }
}

impl YamlConverter {
    /// Convert YAML value to PhasePacket
    fn yaml_value_to_packet(
        &self,
        value: serde_json::Value,
        sequence: usize,
        use_case: &str,
    ) -> EnterpriseResult<PhasePacket> {
        let packet_id = PacketId::new_v4();
        let component_id = ComponentId::new_v4();
        let timestamp = hardware_timestamp();

        let mut metadata = HashMap::new();
        metadata.insert("use_case".to_string(), serde_json::Value::String(use_case.to_string()));
        metadata.insert("original_format".to_string(), serde_json::Value::String("yaml".to_string()));
        metadata.insert("conversion_timestamp".to_string(), serde_json::Value::String(timestamp.to_string()));

        // Extract data from value
        if let serde_json::Value::Object(obj) = &value {
            for (key, val) in obj {
                metadata.insert(key.clone(), val.clone());
            }
        }

        let data = serde_json::to_vec(&value).map_err(|e| {
            EnterpriseError::ConversionFailed {
                format: format!("Failed to serialize YAML data: {}", e),
            }
        })?;

        Ok(PhasePacket {
            header: PhasePacketHeader {
                packet_id,
                sequence: sequence as u64,
                timestamp,
                source_id: component_id,
                packet_type: PacketType::Data,
                flags: PacketFlags::empty(),
                priority: 5,
            },
            payload: PacketPayload {
                data,
                metadata,
            },
        })
    }
}

/// Binary converter implementation
pub struct BinaryConverter;

impl BinaryConverter {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl DataConverter for BinaryConverter {
    async fn convert(&self, context: &ConversionContext) -> EnterpriseResult<Vec<PhasePacket>> {
        let data = tokio::fs::read(&context.file_path).await.map_err(|e| {
            EnterpriseError::FileProcessing {
                reason: format!("Failed to read binary file: {}", e),
            }
        })?;

        let packet_id = PacketId::new_v4();
        let component_id = ComponentId::new_v4();
        let timestamp = hardware_timestamp();

        let mut metadata = HashMap::new();
        metadata.insert("use_case".to_string(), serde_json::Value::String(context.use_case.clone()));
        metadata.insert("original_format".to_string(), serde_json::Value::String("binary".to_string()));
        metadata.insert("conversion_timestamp".to_string(), serde_json::Value::String(timestamp.to_string()));
        metadata.insert("file_size".to_string(), serde_json::Value::Number(serde_json::Number::from(data.len())));

        let packet = PhasePacket {
            header: PhasePacketHeader {
                packet_id,
                sequence: 0,
                timestamp,
                source_id: component_id,
                packet_type: PacketType::Binary,
                flags: PacketFlags::empty(),
                priority: 5,
            },
            payload: PacketPayload {
                data,
                metadata,
            },
        };

        Ok(vec![packet])
    }

    fn supports_format(&self, format: &str) -> bool {
        matches!(format, "binary" | "bin" | "dat")
    }

    fn get_format_name(&self) -> &str {
        "binary"
    }
}
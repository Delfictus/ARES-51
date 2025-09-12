//! Property tests for serialization.

// We need to enable the `net` and `proptest` features for this test.
#![cfg(all(feature = "net", feature = "proptest"))]

use bytes::Bytes;
use csf_core::envelope::Envelope;
use csf_core::error::Error;
use proptest::prelude::*;
use uuid::Uuid;

/// A newtype wrapper around `Envelope` to satisfy the orphan rule.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TestEnvelope(Envelope);

// Implement Arbitrary for our newtype.
impl Arbitrary for TestEnvelope {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        (any::<Vec<u8>>(), any::<[u8; 16]>()) // (payload, uuid_bytes)
            .prop_map(|(payload_vec, uuid_bytes)| {
                let id = Uuid::from_bytes(uuid_bytes);
                let payload = Bytes::from(payload_vec);
                TestEnvelope(Envelope::new_with_id(id, payload))
            })
            .boxed()
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Ensures that an envelope can be serialized and deserialized back to its
    /// original form using bincode.
    #[test]
    fn bincode_roundtrip(test_envelope in any::<TestEnvelope>()) {
        let encoded = bincode::serialize(&test_envelope.0)
            .map_err(|e| Error::Serialization(e.to_string())).unwrap();

        let decoded: Envelope = bincode::deserialize(&encoded)
            .map_err(|e| Error::Serialization(e.to_string())).unwrap();

        prop_assert_eq!(test_envelope.0, decoded);
    }
}

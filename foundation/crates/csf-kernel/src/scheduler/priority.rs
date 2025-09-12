//! Priority calculation for tasks

use crate::task::Task;
use crate::time::hardware_clock;
use csf_core::prelude::*;
use std::cmp::Ordering;

/// Task priority for scheduling decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskPriority {
    /// Static priority level
    pub static_priority: Priority,

    /// Deadline in nanoseconds (earlier = higher priority)
    pub deadline_ns: Option<NanoTime>,

    /// Laxity (slack time) in nanoseconds
    pub laxity_ns: i64,

    /// Age of the task (for aging/starvation prevention)
    pub age_ns: u64,
}

impl TaskPriority {
    /// Create priority from a task
    pub fn from(task: &Task) -> Self {
        let now = hardware_clock::now();

        let laxity_ns = if let Some(deadline) = task.deadline_ns() {
            (deadline.as_nanos() as i64)
                - (now.as_nanos() as i64)
                - (task.estimated_duration_ns() as i64)
        } else {
            i64::MAX // No deadline = maximum laxity
        };

        Self {
            static_priority: task.priority(),
            deadline_ns: task.deadline_ns(),
            laxity_ns,
            age_ns: now.saturating_sub(task.created_at_ns()).as_nanos(),
        }
    }

    /// Update priority based on current time
    pub fn update(&mut self, now: NanoTime) {
        if let Some(deadline) = self.deadline_ns {
            self.laxity_ns = (deadline.as_nanos() as i64) - (now.as_nanos() as i64);
        }
    }
}

impl Ord for TaskPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        // First, compare static priority (lower value = higher priority)
        match self.static_priority.cmp(&other.static_priority) {
            Ordering::Equal => {}
            ord => return ord.reverse(),
        }

        // Then, compare laxity (less laxity = higher priority)
        match self.laxity_ns.cmp(&other.laxity_ns) {
            Ordering::Equal => {}
            ord => return ord.reverse(), // Lower laxity is higher priority, so reverse
        }

        // Finally, compare age (older = higher priority)
        self.age_ns.cmp(&other.age_ns)
    }
}

impl PartialOrd for TaskPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_priority_ordering() {
        let p1 = TaskPriority {
            static_priority: Priority::High,
            deadline_ns: Some(1000.into()),
            laxity_ns: 500,
            age_ns: 10,
        };

        let p2 = TaskPriority {
            static_priority: Priority::Normal,
            deadline_ns: Some(1000.into()),
            laxity_ns: 500,
            age_ns: 10,
        };

        // High priority should be higher than Normal (High = 0, Normal = 1 in enum order)
        assert!(p1 > p2);
        assert_eq!(p1.cmp(&p2), Ordering::Greater);
    }

    #[test]
    fn test_laxity_priority_ordering() {
        // p1 has less slack time (is more urgent), so it's higher priority
        let p1 = TaskPriority {
            static_priority: Priority::Normal,
            deadline_ns: Some(1000.into()),
            laxity_ns: 100,
            age_ns: 10,
        };
        let p2 = TaskPriority {
            static_priority: Priority::Normal,
            deadline_ns: Some(2000.into()),
            laxity_ns: 200,
            age_ns: 10,
        };
        assert!(p1 > p2);
        assert_eq!(p1.cmp(&p2), Ordering::Greater);
    }

    #[test]
    fn test_age_priority_ordering() {
        // p2 is older, so it's higher priority (as a tie-breaker)
        let p1 = TaskPriority {
            static_priority: Priority::Normal,
            deadline_ns: Some(1000.into()),
            laxity_ns: 100,
            age_ns: 10,
        };
        let p2 = TaskPriority {
            static_priority: Priority::Normal,
            deadline_ns: Some(1000.into()),
            laxity_ns: 100,
            age_ns: 20,
        };
        assert!(p2 > p1);
        assert_eq!(p2.cmp(&p1), Ordering::Greater);
    }
}

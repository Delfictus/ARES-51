use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub id: String,
    pub initial_balance: Decimal,
    pub current_balance: Decimal,
    pub available_balance: Decimal,
    pub margin_used: Decimal,
    pub realized_pnl: Decimal,
    pub unrealized_pnl: Decimal,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub is_active: bool,
    pub leverage: Decimal,
    pub margin_call_level: Decimal,
    pub stop_out_level: Decimal,
}

impl Account {
    pub fn new(id: String, initial_balance: Decimal) -> Self {
        let now = Utc::now();
        Self {
            id,
            initial_balance,
            current_balance: initial_balance,
            available_balance: initial_balance,
            margin_used: dec!(0),
            realized_pnl: dec!(0),
            unrealized_pnl: dec!(0),
            created_at: now,
            updated_at: now,
            is_active: true,
            leverage: dec!(1),
            margin_call_level: dec!(0.5),  // 50% margin call
            stop_out_level: dec!(0.2),      // 20% stop out
        }
    }

    pub fn equity(&self) -> Decimal {
        self.current_balance + self.unrealized_pnl
    }

    pub fn margin_level(&self) -> Option<Decimal> {
        if self.margin_used > dec!(0) {
            Some((self.equity() / self.margin_used) * dec!(100))
        } else {
            None
        }
    }

    pub fn free_margin(&self) -> Decimal {
        self.equity() - self.margin_used
    }

    pub fn add_funds(&mut self, amount: Decimal) -> Result<()> {
        if amount <= dec!(0) {
            return Err(anyhow::anyhow!("Amount must be positive"));
        }
        self.current_balance += amount;
        self.available_balance += amount;
        self.updated_at = Utc::now();
        Ok(())
    }

    pub fn withdraw_funds(&mut self, amount: Decimal) -> Result<()> {
        if amount <= dec!(0) {
            return Err(anyhow::anyhow!("Amount must be positive"));
        }
        if amount > self.available_balance {
            return Err(anyhow::anyhow!("Insufficient available balance"));
        }
        self.current_balance -= amount;
        self.available_balance -= amount;
        self.updated_at = Utc::now();
        Ok(())
    }

    pub fn update_pnl(&mut self, realized: Decimal, unrealized: Decimal) {
        self.realized_pnl = realized;
        self.unrealized_pnl = unrealized;
        self.updated_at = Utc::now();
    }

    pub fn lock_margin(&mut self, amount: Decimal) -> Result<()> {
        if amount > self.available_balance {
            return Err(anyhow::anyhow!("Insufficient available balance for margin"));
        }
        self.margin_used += amount;
        self.available_balance -= amount;
        self.updated_at = Utc::now();
        Ok(())
    }

    pub fn release_margin(&mut self, amount: Decimal) -> Result<()> {
        if amount > self.margin_used {
            return Err(anyhow::anyhow!("Cannot release more margin than used"));
        }
        self.margin_used -= amount;
        self.available_balance += amount;
        self.updated_at = Utc::now();
        Ok(())
    }

    pub fn is_margin_call(&self) -> bool {
        if let Some(level) = self.margin_level() {
            level < self.margin_call_level * dec!(100)
        } else {
            false
        }
    }

    pub fn is_stop_out(&self) -> bool {
        if let Some(level) = self.margin_level() {
            level < self.stop_out_level * dec!(100)
        } else {
            false
        }
    }
}

pub struct AccountManager {
    accounts: Arc<RwLock<Vec<Account>>>,
    active_account: Arc<RwLock<Option<String>>>,
}

impl AccountManager {
    pub fn new() -> Self {
        Self {
            accounts: Arc::new(RwLock::new(Vec::new())),
            active_account: Arc::new(RwLock::new(None)),
        }
    }

    pub async fn create_account(&self, id: String, initial_balance: Decimal) -> Result<Account> {
        let account = Account::new(id.clone(), initial_balance);
        let mut accounts = self.accounts.write().await;
        accounts.push(account.clone());
        
        // Set as active if it's the first account
        if accounts.len() == 1 {
            let mut active = self.active_account.write().await;
            *active = Some(id);
        }
        
        Ok(account)
    }

    pub async fn get_account(&self, id: &str) -> Result<Option<Account>> {
        let accounts = self.accounts.read().await;
        Ok(accounts.iter().find(|a| a.id == id).cloned())
    }

    pub async fn get_active_account(&self) -> Result<Option<Account>> {
        let active_id = self.active_account.read().await;
        if let Some(id) = active_id.as_ref() {
            self.get_account(id).await
        } else {
            Ok(None)
        }
    }

    pub async fn set_active_account(&self, id: String) -> Result<()> {
        let accounts = self.accounts.read().await;
        if !accounts.iter().any(|a| a.id == id) {
            return Err(anyhow::anyhow!("Account not found"));
        }
        let mut active = self.active_account.write().await;
        *active = Some(id);
        Ok(())
    }

    pub async fn update_account<F>(&self, id: &str, update_fn: F) -> Result<()>
    where
        F: FnOnce(&mut Account) -> Result<()>,
    {
        let mut accounts = self.accounts.write().await;
        let account = accounts
            .iter_mut()
            .find(|a| a.id == id)
            .ok_or_else(|| anyhow::anyhow!("Account not found"))?;
        update_fn(account)?;
        Ok(())
    }

    pub async fn get_total_equity(&self) -> Decimal {
        let accounts = self.accounts.read().await;
        accounts.iter().map(|a| a.equity()).sum()
    }

    pub async fn get_all_accounts(&self) -> Vec<Account> {
        self.accounts.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_account_creation() {
        let manager = AccountManager::new();
        let account = manager.create_account(
            "test-account".to_string(),
            dec!(10000)
        ).await.unwrap();
        
        assert_eq!(account.initial_balance, dec!(10000));
        assert_eq!(account.current_balance, dec!(10000));
        assert_eq!(account.available_balance, dec!(10000));
        assert_eq!(account.margin_used, dec!(0));
    }

    #[tokio::test]
    async fn test_margin_operations() {
        let manager = AccountManager::new();
        let account_id = "test-account".to_string();
        manager.create_account(account_id.clone(), dec!(10000)).await.unwrap();
        
        // Lock margin
        manager.update_account(&account_id, |acc| {
            acc.lock_margin(dec!(2000))
        }).await.unwrap();
        
        let account = manager.get_account(&account_id).await.unwrap().unwrap();
        assert_eq!(account.margin_used, dec!(2000));
        assert_eq!(account.available_balance, dec!(8000));
        
        // Release margin
        manager.update_account(&account_id, |acc| {
            acc.release_margin(dec!(1000))
        }).await.unwrap();
        
        let account = manager.get_account(&account_id).await.unwrap().unwrap();
        assert_eq!(account.margin_used, dec!(1000));
        assert_eq!(account.available_balance, dec!(9000));
    }
}
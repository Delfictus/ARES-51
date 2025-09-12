use sqlx::{SqlitePool, sqlite::SqlitePoolOptions};
use rust_decimal::Decimal;
use chrono::{DateTime, Utc};
use anyhow::Result;

use crate::orders::Order;
use crate::portfolio::Position;

pub struct Database {
    pool: SqlitePool,
}

impl Database {
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await?;
        
        Ok(Self { pool })
    }

    pub async fn init_schema(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                quantity DECIMAL NOT NULL,
                price DECIMAL,
                status TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                filled_at TIMESTAMP,
                filled_price DECIMAL,
                commission DECIMAL
            )
            "#
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                quantity DECIMAL NOT NULL,
                average_price DECIMAL NOT NULL,
                current_price DECIMAL NOT NULL,
                unrealized_pnl DECIMAL NOT NULL,
                realized_pnl DECIMAL NOT NULL,
                opened_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
            "#
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity DECIMAL NOT NULL,
                price DECIMAL NOT NULL,
                pnl DECIMAL,
                timestamp TIMESTAMP NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders(id)
            )
            "#
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS account_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                balance DECIMAL NOT NULL,
                equity DECIMAL NOT NULL,
                margin_used DECIMAL NOT NULL,
                realized_pnl DECIMAL NOT NULL,
                unrealized_pnl DECIMAL NOT NULL,
                timestamp TIMESTAMP NOT NULL
            )
            "#
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn save_order(&self, order: &Order) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO orders (
                id, symbol, side, order_type, quantity, price, 
                status, created_at, filled_at, filled_price, commission
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#
        )
        .bind(&order.id)
        .bind(&order.symbol)
        .bind(format!("{:?}", order.side))
        .bind(format!("{:?}", order.order_type))
        .bind(order.quantity)
        .bind(order.price)
        .bind(format!("{:?}", order.status))
        .bind(order.created_at)
        .bind(order.filled_at)
        .bind(order.filled_price)
        .bind(order.commission)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }

    pub async fn save_position(&self, position: &Position) -> Result<()> {
        sqlx::query(
            r#"
            INSERT OR REPLACE INTO positions (
                symbol, quantity, average_price, current_price,
                unrealized_pnl, realized_pnl, opened_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            "#
        )
        .bind(&position.symbol)
        .bind(position.quantity)
        .bind(position.average_price)
        .bind(position.current_price)
        .bind(position.unrealized_pnl)
        .bind(position.realized_pnl)
        .bind(position.opened_at)
        .bind(position.updated_at)
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }

    pub async fn get_historical_orders(&self, limit: i64) -> Result<Vec<Order>> {
        // Implementation would fetch and deserialize orders
        Ok(Vec::new())
    }

    pub async fn get_historical_positions(&self) -> Result<Vec<Position>> {
        // Implementation would fetch and deserialize positions
        Ok(Vec::new())
    }
}
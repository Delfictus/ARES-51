use rust_decimal::Decimal;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: Decimal,
    pub price: Option<Decimal>,
    pub stop_loss: Option<Decimal>,
    pub take_profit: Option<Decimal>,
    pub status: OrderStatus,
    pub created_at: DateTime<Utc>,
    pub filled_at: Option<DateTime<Utc>>,
    pub filled_price: Option<Decimal>,
    pub commission: Decimal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum OrderStatus {
    Pending,
    Submitted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

impl Order {
    pub fn new_market_order(symbol: String, side: OrderSide, quantity: Decimal) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            symbol,
            side,
            order_type: OrderType::Market,
            quantity,
            price: None,
            stop_loss: None,
            take_profit: None,
            status: OrderStatus::Pending,
            created_at: Utc::now(),
            filled_at: None,
            filled_price: None,
            commission: rust_decimal_macros::dec!(0),
        }
    }

    pub fn new_limit_order(
        symbol: String,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            symbol,
            side,
            order_type: OrderType::Limit,
            quantity,
            price: Some(price),
            stop_loss: None,
            take_profit: None,
            status: OrderStatus::Pending,
            created_at: Utc::now(),
            filled_at: None,
            filled_price: None,
            commission: rust_decimal_macros::dec!(0),
        }
    }

    pub fn is_open(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Pending | OrderStatus::Submitted | OrderStatus::PartiallyFilled
        )
    }

    pub fn is_filled(&self) -> bool {
        self.status == OrderStatus::Filled
    }

    pub fn total_value(&self) -> Decimal {
        let price = self.filled_price.or(self.price).unwrap_or(rust_decimal_macros::dec!(0));
        price * self.quantity
    }
}

pub struct OrderManager {
    orders: Arc<RwLock<Vec<Order>>>,
    order_counter: Arc<RwLock<u64>>,
}

impl OrderManager {
    pub fn new() -> Self {
        Self {
            orders: Arc::new(RwLock::new(Vec::new())),
            order_counter: Arc::new(RwLock::new(0)),
        }
    }

    pub async fn submit_order(&self, mut order: Order) -> Result<Order> {
        order.status = OrderStatus::Submitted;
        let mut orders = self.orders.write().await;
        orders.push(order.clone());
        Ok(order)
    }

    pub async fn cancel_order(&self, order_id: &str) -> Result<()> {
        let mut orders = self.orders.write().await;
        if let Some(order) = orders.iter_mut().find(|o| o.id == order_id) {
            if order.is_open() {
                order.status = OrderStatus::Cancelled;
                Ok(())
            } else {
                Err(anyhow::anyhow!("Cannot cancel order in status: {:?}", order.status))
            }
        } else {
            Err(anyhow::anyhow!("Order not found"))
        }
    }

    pub async fn get_order(&self, order_id: &str) -> Option<Order> {
        let orders = self.orders.read().await;
        orders.iter().find(|o| o.id == order_id).cloned()
    }

    pub async fn get_open_orders(&self) -> Vec<Order> {
        let orders = self.orders.read().await;
        orders.iter().filter(|o| o.is_open()).cloned().collect()
    }

    pub async fn get_filled_orders(&self) -> Vec<Order> {
        let orders = self.orders.read().await;
        orders.iter().filter(|o| o.is_filled()).cloned().collect()
    }

    pub async fn get_orders_by_symbol(&self, symbol: &str) -> Vec<Order> {
        let orders = self.orders.read().await;
        orders.iter().filter(|o| o.symbol == symbol).cloned().collect()
    }

    pub async fn update_order_status(
        &self,
        order_id: &str,
        status: OrderStatus,
        filled_price: Option<Decimal>,
    ) -> Result<()> {
        let mut orders = self.orders.write().await;
        if let Some(order) = orders.iter_mut().find(|o| o.id == order_id) {
            order.status = status;
            if status == OrderStatus::Filled {
                order.filled_at = Some(Utc::now());
                order.filled_price = filled_price;
                
                // Calculate commission (0.1% of trade value)
                if let Some(price) = filled_price {
                    order.commission = price * order.quantity * rust_decimal_macros::dec!(0.001);
                }
            }
            Ok(())
        } else {
            Err(anyhow::anyhow!("Order not found"))
        }
    }

    pub async fn get_all_orders(&self) -> Vec<Order> {
        self.orders.read().await.clone()
    }

    pub async fn clear_orders(&self) {
        self.orders.write().await.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_order_management() {
        let manager = OrderManager::new();
        
        let order = Order::new_market_order(
            "AAPL".to_string(),
            OrderSide::Buy,
            rust_decimal_macros::dec!(100),
        );
        
        let submitted = manager.submit_order(order).await.unwrap();
        assert_eq!(submitted.status, OrderStatus::Submitted);
        
        // Update to filled
        manager.update_order_status(
            &submitted.id,
            OrderStatus::Filled,
            Some(rust_decimal_macros::dec!(175.50)),
        ).await.unwrap();
        
        let filled = manager.get_order(&submitted.id).await.unwrap();
        assert_eq!(filled.status, OrderStatus::Filled);
        assert_eq!(filled.filled_price, Some(rust_decimal_macros::dec!(175.50)));
        assert!(filled.commission > rust_decimal_macros::dec!(0));
    }
}

use uuid;
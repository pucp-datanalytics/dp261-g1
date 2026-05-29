# Diccionario de Feature Engineering

Este documento resume las variables creadas por `src/preprocessing.py`. Las mismas transformaciones se ejecutan en notebooks, entrenamiento, API y dashboard porque están encapsuladas en `FeatureEngineeringTransformer`.

| Feature | Cálculo | Hipótesis de negocio |
|---|---|---|
| `PurchYear` | Año de `PurchDate` | Captura cambios del mercado por periodo. |
| `PurchMonth` | Mes de `PurchDate` | Captura estacionalidad de subastas. |
| `PurchQuarter` | Trimestre de `PurchDate` | Resume estacionalidad con menor granularidad. |
| `PurchDayOfWeek` | Día de semana de `PurchDate` | Puede capturar diferencias operativas por día de compra. |
| `odo_per_year` | `VehOdo / (VehicleAge + 1)` | Mide intensidad de uso anual. |
| `old_high_mileage_flag` | Flag de edad >= 5 y odómetro >= 85,000 | Marca doble señal de desgaste. |
| `cost_to_acq_auction_avg` | `VehBCost / MMRAcquisitionAuctionAveragePrice` | Detecta sobrepago/infrapago vs. mercado de adquisición. |
| `acq_auction_margin` | `MMRAcquisitionAuctionAveragePrice - VehBCost` | Estima colchón económico al adquirir. |
| `cost_to_current_auction_avg` | `VehBCost / MMRCurrentAuctionAveragePrice` | Compara costo con valor actual de subasta. |
| `current_auction_margin` | `MMRCurrentAuctionAveragePrice - VehBCost` | Estima margen actual de mercado. |
| `warranty_to_cost` | `WarrantyCost / VehBCost` | Garantía alta relativa al costo puede indicar riesgo mecánico. |
| `warranty_per_vehicle_year` | `WarrantyCost / (VehicleAge + 1)` | Ajusta garantía por antigüedad del vehículo. |
| `auction_avg_depreciation` | `MMRCurrentAuctionAveragePrice - MMRAcquisitionAuctionAveragePrice` | Cambio de valor de mercado de subasta. |
| `retail_avg_depreciation` | `MMRCurrentRetailAveragePrice - MMRAcquisitionRetailAveragePrice` | Cambio de valor retail estimado. |
| `acq_clean_avg_spread` | `MMRAcquisitionAuctionCleanPrice - MMRAcquisitionAuctionAveragePrice` | Diferencia entre condición clean y promedio al adquirir. |
| `current_clean_avg_spread` | `MMRCurrentAuctionCleanPrice - MMRCurrentAuctionAveragePrice` | Diferencia actual entre condición clean y promedio. |
| `mmr_missing_count` | Conteo de precios MMR faltantes | La ausencia de información puede ser una señal de incertidumbre. |

## Nota de diseño

No se usa el target para crear ninguna variable. Por tanto, estas features pueden calcularse tanto en train como en producción sin data leakage.

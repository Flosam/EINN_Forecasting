from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariableSpec:
    name: str
    candidates: list[str]
    prefer_seasonally_adjusted: bool = True
    interpolation_method: str | None = None
    aggregation_method: str = "last"


VARIABLE_SPECS: list[VariableSpec] = [
    VariableSpec("cpi_all_items", ["CPIAUCSL"], True),
    VariableSpec("unemployment_rate", ["UNRATE"], True, aggregation_method="mean"),
    VariableSpec("natural_rate_unemployment", ["NROU"], False, interpolation_method="linear"),
    VariableSpec("inflation_expectations_umich", ["MICH"], False, aggregation_method="mean"),
    VariableSpec("industrial_production", ["INDPRO"], True),
    VariableSpec("wti_oil_price", ["WTISPLC"], False),
    VariableSpec("producer_price_index", ["PPIACO"], True),
    VariableSpec("fed_funds_rate", ["FEDFUNDS"], False, aggregation_method="mean"),
    VariableSpec("financial_conditions", ["NFCI", "ANFCI"], False),
    VariableSpec("payroll_employment", ["PAYEMS"], True),
    VariableSpec("real_personal_income_less_transfers", ["W875RX1"], True),
    VariableSpec("housing_starts", ["HOUST"], True),
    VariableSpec("retail_sales", ["RSAFS"], True),
    VariableSpec("capacity_utilization", ["TCU"], True, aggregation_method="mean"),
    VariableSpec("real_disposable_personal_income", ["DSPIC96"], True),
    VariableSpec("money_stock_m2", ["M2SL"], False),
    VariableSpec("treasury_10y", ["GS10"], False, aggregation_method="mean"),
    VariableSpec("treasury_3m", ["TB3MS"], False, aggregation_method="mean"),
]


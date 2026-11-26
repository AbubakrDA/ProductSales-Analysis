from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Product sales by region",
    page_icon=":material/analytics:",
    layout="wide",
)

DATA_FILE = Path(__file__).resolve().parent / "Product-Sales-Region.xlsx"


@st.cache_data(ttl=3600, show_spinner="Loading sales data...")
def load_sales_data(path: Path) -> pd.DataFrame:
    """Load, clean, and engineer features matching Product_Sales_Analysis.ipynb."""
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()

    text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in text_cols:
        df[col] = df[col].str.strip()

    for col in ["Date", "OrderDate", "DeliveryDate"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.drop_duplicates(subset="OrderID", keep="first").reset_index(drop=True)
    df["Promotion"] = df["Promotion"].fillna("None")

    if "Date" in df.columns and (df["Date"] == df["OrderDate"]).all():
        df = df.drop(columns=["Date"])

    df["Year"] = df["OrderDate"].dt.year
    df["Month"] = df["OrderDate"].dt.month
    df["Quarter"] = df["OrderDate"].dt.to_period("Q").astype(str)
    df["YearMonth"] = df["OrderDate"].dt.to_period("M").dt.to_timestamp()
    df["Half"] = df["Year"].astype(str) + "-H" + np.where(df["Month"] <= 6, "1", "2")

    df["GrossSales"] = df["Quantity"] * df["UnitPrice"]
    df["DiscountAmount"] = df["GrossSales"] - df["TotalPrice"]
    df["DeliveryDays"] = (df["DeliveryDate"] - df["OrderDate"]).dt.days
    df["NetRevenue"] = np.where(df["Returned"] == 1, 0.0, df["TotalPrice"])
    df["ReturnedRevenue"] = np.where(df["Returned"] == 1, df["TotalPrice"], 0.0)
    df["DiscountPct"] = (df["Discount"] * 100).round(0).astype(int).astype(str) + "%"
    return df


if not DATA_FILE.exists():
    st.error(
        f"Could not find `{DATA_FILE.name}` next to `streamlit_app.py`.",
        title="Dataset not found",
        icon=":material/error:",
    )
    st.stop()

raw_df = load_sales_data(DATA_FILE)

# Precompute filter option lists
min_date = raw_df["OrderDate"].min().date()
max_date = raw_df["OrderDate"].max().date()
regions_all = sorted(raw_df["Region"].unique().tolist())
products_all = sorted(raw_df["Product"].unique().tolist())
salespeople_all = sorted(raw_df["Salesperson"].unique().tolist())
payments_all = sorted(raw_df["PaymentMethod"].unique().tolist())
promotions_all = sorted(raw_df["Promotion"].unique().tolist())
stores_all = sorted(raw_df["StoreLocation"].unique().tolist())
discounts_all = [f"{d:.0%}" for d in sorted(raw_df["Discount"].unique().tolist())]

FILTER_DEFAULTS = {
    "filter_dates": (min_date, max_date),
    "filter_regions": regions_all,
    "filter_products": products_all,
    "filter_customer_type": "All",
    "filter_returned": "All",
    "filter_salespeople": salespeople_all,
    "filter_payments": payments_all,
    "filter_promotions": promotions_all,
    "filter_stores": stores_all,
    "filter_discounts": discounts_all,
}

for key, default_val in FILTER_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_val


def reset_filters() -> None:
    """Restore all sidebar slicers to their default values."""
    for k, v in FILTER_DEFAULTS.items():
        st.session_state[k] = v


# ---- Sidebar slicers & filters ----
with st.sidebar:
    st.header("Slicers & filters", divider="gray")
    st.button(
        "Reset all filters",
        icon=":material/refresh:",
        width="stretch",
        on_click=reset_filters,
    )

    date_range = st.date_input(
        "Order date range",
        min_value=min_date,
        max_value=max_date,
        key="filter_dates",
    )
    selected_regions = st.pills(
        "Region",
        options=regions_all,
        selection_mode="multi",
        key="filter_regions",
    )
    selected_products = st.multiselect(
        "Product",
        options=products_all,
        key="filter_products",
    )
    selected_customer_type = st.segmented_control(
        "Customer type",
        options=["All", "Retail", "Wholesale"],
        key="filter_customer_type",
    )
    selected_returned = st.segmented_control(
        "Return status",
        options=["All", "Not returned", "Returned"],
        key="filter_returned",
    )

    with st.expander("More filters", expanded=False):
        selected_salespeople = st.multiselect(
            "Salesperson",
            options=salespeople_all,
            key="filter_salespeople",
        )
        selected_payments = st.pills(
            "Payment method",
            options=payments_all,
            selection_mode="multi",
            key="filter_payments",
        )
        selected_promotions = st.pills(
            "Promotion",
            options=promotions_all,
            selection_mode="multi",
            key="filter_promotions",
        )
        selected_stores = st.pills(
            "Store location",
            options=stores_all,
            selection_mode="multi",
            key="filter_stores",
        )
        selected_discounts = st.pills(
            "Discount rate",
            options=discounts_all,
            selection_mode="multi",
            key="filter_discounts",
        )

# Handle partial date selection while user is picking start/end date
if isinstance(date_range, (tuple, list)):
    if len(date_range) == 2:
        start_date, end_date = date_range
    elif len(date_range) == 1:
        start_date = end_date = date_range[0]
    else:
        start_date, end_date = min_date, max_date
else:
    start_date = end_date = date_range

# Apply all active filters
mask = (
    (raw_df["OrderDate"].dt.date >= start_date)
    & (raw_df["OrderDate"].dt.date <= end_date)
    & (raw_df["Region"].isin(selected_regions))
    & (raw_df["Product"].isin(selected_products))
    & (raw_df["Salesperson"].isin(selected_salespeople))
    & (raw_df["PaymentMethod"].isin(selected_payments))
    & (raw_df["Promotion"].isin(selected_promotions))
    & (raw_df["StoreLocation"].isin(selected_stores))
    & (raw_df["DiscountPct"].isin(selected_discounts))
)

if selected_customer_type and selected_customer_type != "All":
    mask &= raw_df["CustomerType"] == selected_customer_type

if selected_returned == "Returned":
    mask &= raw_df["Returned"] == 1
elif selected_returned == "Not returned":
    mask &= raw_df["Returned"] == 0

df = raw_df.loc[mask].copy()

# ---- Header ----
header_col, action_col = st.columns([4, 1], vertical_alignment="bottom")
with header_col:
    st.title("Product sales by region")
    st.caption(
        f"Showing **{len(df):,}** of {len(raw_df):,} orders "
        f"({len(df) / len(raw_df):.1%}) · {start_date} to {end_date}"
    )
with action_col:
    st.download_button(
        "Export filtered CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_product_sales.csv",
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
        disabled=df.empty,
    )

if df.empty:
    st.warning(
        "No orders match the current filter selection. Try widening your filters or click **Reset all filters** in the sidebar.",
        title="No matching orders",
        icon=":material/filter_alt_off:",
    )
    st.stop()


def fmt_currency(val: float) -> str:
    if abs(val) >= 1e6:
        return f"${val / 1e6:,.2f}M"
    if abs(val) >= 1e3:
        return f"${val / 1e3:,.1f}K"
    return f"${val:,.0f}"


# ---- Row 0: KPI tiles ----
kpi_cols = st.columns(6)
kpis = [
    ("Gross revenue", fmt_currency(df["TotalPrice"].sum())),
    ("Net revenue", fmt_currency(df["NetRevenue"].sum())),
    ("Returned revenue", fmt_currency(df["ReturnedRevenue"].sum())),
    ("Return rate", f"{df['Returned'].mean():.1%}"),
    ("Avg order value", f"${df['TotalPrice'].mean():,.0f}"),
    ("Orders / customers", f"{len(df):,} / {df['CustomerName'].nunique():,}"),
]
for col, (label, val) in zip(kpi_cols, kpis):
    col.metric(label, val, border=True)

overall_rr = df["Returned"].mean() * 100

# ---- Row 1: Region, Region x Product, Monthly trend ----
r1_c1, r1_c2, r1_c3 = st.columns(3)

with r1_c1:
    with st.container(border=True):
        st.subheader("A. Revenue by region ($K)")
        reg = (
            df.groupby("Region", as_index=False)
            .agg(Gross=("TotalPrice", "sum"), Net=("NetRevenue", "sum"), RR=("Returned", "mean"))
        )
        reg["GrossK"] = reg["Gross"] / 1e3
        reg["NetK"] = reg["Net"] / 1e3
        reg_long = reg.melt(
            id_vars=["Region", "RR"],
            value_vars=["GrossK", "NetK"],
            var_name="Metric",
            value_name="RevenueK",
        )
        reg_long["Metric"] = reg_long["Metric"].map(
            {"GrossK": "Gross", "NetK": "Net (after returns)"}
        )
        chart_a = (
            alt.Chart(reg_long)
            .mark_bar()
            .encode(
                y=alt.Y("Region:N", sort="-x", title=None),
                x=alt.X("RevenueK:Q", title="$ thousands"),
                color=alt.Color(
                    "Metric:N",
                    scale=alt.Scale(
                        domain=["Gross", "Net (after returns)"],
                        range=["#4c72b0", "#55a868"],
                    ),
                    legend=alt.Legend(orient="bottom", title=None),
                ),
                yOffset="Metric:N",
                tooltip=[
                    alt.Tooltip("Region:N"),
                    alt.Tooltip("Metric:N"),
                    alt.Tooltip("RevenueK:Q", title="Revenue ($K)", format=",.1f"),
                    alt.Tooltip("RR:Q", title="Return rate", format=".1%"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(chart_a, width="stretch", alt="Revenue by region bar chart")

with r1_c2:
    with st.container(border=True):
        st.subheader("B. Region × product ($K)")
        hm = (
            df.groupby(["Region", "Product"], as_index=False)
            .agg(Revenue=("TotalPrice", "sum"))
        )
        hm["RevenueK"] = hm["Revenue"] / 1e3
        base_hm = alt.Chart(hm).encode(
            x=alt.X("Product:N", title=None),
            y=alt.Y("Region:N", title=None),
        )
        rect = base_hm.mark_rect().encode(
            color=alt.Color("RevenueK:Q", scale=alt.Scale(scheme="blues"), title="$K"),
            tooltip=[
                alt.Tooltip("Region:N"),
                alt.Tooltip("Product:N"),
                alt.Tooltip("RevenueK:Q", title="Revenue ($K)", format=",.1f"),
            ],
        )
        text = base_hm.mark_text(fontSize=10).encode(
            text=alt.Text("RevenueK:Q", format=".0f"),
            color=alt.condition(
                alt.datum.RevenueK > hm["RevenueK"].max() * 0.55,
                alt.value("white"),
                alt.value("#1f2937"),
            ),
        )
        st.altair_chart(
            (rect + text).properties(height=280),
            width="stretch",
            alt="Revenue heatmap by region and product",
        )

with r1_c3:
    with st.container(border=True):
        st.subheader("C. Monthly revenue trend ($K)")
        monthly = (
            df.groupby("YearMonth", as_index=False)["TotalPrice"]
            .sum()
            .sort_values("YearMonth")
        )
        monthly["MonthlyK"] = monthly["TotalPrice"] / 1e3
        monthly["Rolling3K"] = monthly["MonthlyK"].rolling(3, min_periods=1).mean()
        monthly_long = monthly.melt(
            id_vars=["YearMonth"],
            value_vars=["MonthlyK", "Rolling3K"],
            var_name="Series",
            value_name="RevenueK",
        )
        monthly_long["Series"] = monthly_long["Series"].map(
            {"MonthlyK": "Monthly", "Rolling3K": "3-month rolling avg"}
        )
        chart_c = (
            alt.Chart(monthly_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("YearMonth:T", title=None),
                y=alt.Y("RevenueK:Q", title="$ thousands"),
                color=alt.Color(
                    "Series:N",
                    scale=alt.Scale(
                        domain=["Monthly", "3-month rolling avg"],
                        range=["#4c72b0", "#c44e52"],
                    ),
                    legend=alt.Legend(orient="bottom", title=None),
                ),
                strokeWidth=alt.condition(
                    alt.datum.Series == "3-month rolling avg", alt.value(3), alt.value(1.5)
                ),
                tooltip=[
                    alt.Tooltip("YearMonth:T", title="Month", format="%b %Y"),
                    alt.Tooltip("Series:N"),
                    alt.Tooltip("RevenueK:Q", title="Revenue ($K)", format=",.1f"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(chart_c, width="stretch", alt="Monthly revenue trend line chart")

# ---- Row 2: Product, Salesperson, Payment ----
r2_c1, r2_c2, r2_c3 = st.columns(3)

with r2_c1:
    with st.container(border=True):
        st.subheader("D. Revenue by product ($K)")
        prod = (
            df.groupby("Product", as_index=False)
            .agg(Gross=("TotalPrice", "sum"), RR=("Returned", "mean"), Orders=("OrderID", "size"))
        )
        prod["GrossK"] = prod["Gross"] / 1e3
        chart_d = (
            alt.Chart(prod)
            .mark_bar(color="#4c72b0")
            .encode(
                x=alt.X("Product:N", sort="-y", title=None),
                y=alt.Y("GrossK:Q", title="$ thousands"),
                tooltip=[
                    alt.Tooltip("Product:N"),
                    alt.Tooltip("GrossK:Q", title="Gross ($K)", format=",.1f"),
                    alt.Tooltip("RR:Q", title="Return rate", format=".1%"),
                    alt.Tooltip("Orders:Q", format=","),
                ],
            )
            .properties(height=270)
        )
        st.altair_chart(chart_d, width="stretch", alt="Revenue by product bar chart")

with r2_c2:
    with st.container(border=True):
        st.subheader("E. Salesperson avg order value ($)")
        sp = (
            df.groupby("Salesperson", as_index=False)
            .agg(AOV=("TotalPrice", "mean"), Total=("TotalPrice", "sum"), Orders=("OrderID", "size"))
        )
        sp["TotalK"] = sp["Total"] / 1e3
        chart_e = (
            alt.Chart(sp)
            .mark_bar(color="#8172b3")
            .encode(
                y=alt.Y("Salesperson:N", sort="-x", title=None),
                x=alt.X("AOV:Q", title="Average order value ($)"),
                tooltip=[
                    alt.Tooltip("Salesperson:N"),
                    alt.Tooltip("AOV:Q", title="Avg order value ($)", format="$,.0f"),
                    alt.Tooltip("TotalK:Q", title="Total revenue ($K)", format=",.1f"),
                    alt.Tooltip("Orders:Q", format=","),
                ],
            )
            .properties(height=270)
        )
        st.altair_chart(chart_e, width="stretch", alt="Salesperson average order value chart")

with r2_c3:
    with st.container(border=True):
        st.subheader("F. Return rate by payment (%, 95% CI)")
        pay = (
            df.groupby("PaymentMethod", as_index=False)
            .agg(Orders=("Returned", "size"), ReturnRate=("Returned", "mean"))
        )
        pay["CI95"] = 1.96 * np.sqrt(
            pay["ReturnRate"] * (1 - pay["ReturnRate"]) / pay["Orders"].clip(lower=1)
        )
        pay["RatePct"] = pay["ReturnRate"] * 100
        pay["LowPct"] = ((pay["ReturnRate"] - pay["CI95"]).clip(lower=0)) * 100
        pay["HighPct"] = ((pay["ReturnRate"] + pay["CI95"]).clip(upper=1)) * 100

        bars_f = (
            alt.Chart(pay)
            .mark_bar(color="#c44e52")
            .encode(
                x=alt.X("PaymentMethod:N", sort="y", title=None),
                y=alt.Y("RatePct:Q", title="% returned"),
                tooltip=[
                    alt.Tooltip("PaymentMethod:N", title="Payment method"),
                    alt.Tooltip("RatePct:Q", title="Return rate (%)", format=".1f"),
                    alt.Tooltip("Orders:Q", format=","),
                ],
            )
        )
        err_f = (
            alt.Chart(pay)
            .mark_errorbar(color="#1f2937")
            .encode(
                x=alt.X("PaymentMethod:N", sort="y"),
                y=alt.Y("LowPct:Q", title="% returned"),
                y2="HighPct:Q",
            )
        )
        rule_f = (
            alt.Chart(pd.DataFrame({"y": [overall_rr]}))
            .mark_rule(strokeDash=[5, 5], color="gray")
            .encode(y="y:Q")
        )
        st.altair_chart(
            (bars_f + err_f + rule_f).properties(height=270),
            width="stretch",
            alt="Return rate by payment method with 95% confidence intervals",
        )

# ---- Row 3: Promotion, Discount, Delivery ----
r3_c1, r3_c2, r3_c3 = st.columns(3)

with r3_c1:
    with st.container(border=True):
        st.subheader("G. Promotions: AOV ($) & return rate")
        promo_t = (
            df.groupby("Promotion", as_index=False)
            .agg(AOV=("TotalPrice", "mean"), ReturnRate=("Returned", "mean"), Orders=("OrderID", "size"))
        )
        chart_g = (
            alt.Chart(promo_t)
            .mark_bar(color="#55a868")
            .encode(
                x=alt.X("Promotion:N", sort="-y", title=None),
                y=alt.Y("AOV:Q", title="Average order value ($)"),
                tooltip=[
                    alt.Tooltip("Promotion:N"),
                    alt.Tooltip("AOV:Q", title="Avg order value ($)", format="$,.0f"),
                    alt.Tooltip("ReturnRate:Q", title="Return rate", format=".1%"),
                    alt.Tooltip("Orders:Q", format=","),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(chart_g, width="stretch", alt="Promotions average order value chart")

with r3_c2:
    with st.container(border=True):
        st.subheader("H. Avg order value by discount ($)")
        disc_t = (
            df.groupby(["Discount", "DiscountPct"], as_index=False)
            .agg(AOV=("TotalPrice", "mean"), ReturnRate=("Returned", "mean"), Orders=("OrderID", "size"))
            .sort_values("Discount")
        )
        chart_h = (
            alt.Chart(disc_t)
            .mark_bar(color="#4c72b0")
            .encode(
                x=alt.X("DiscountPct:N", sort=disc_t["DiscountPct"].tolist(), title="Discount rate"),
                y=alt.Y("AOV:Q", title="Average order value ($)"),
                tooltip=[
                    alt.Tooltip("DiscountPct:N", title="Discount"),
                    alt.Tooltip("AOV:Q", title="Avg order value ($)", format="$,.0f"),
                    alt.Tooltip("ReturnRate:Q", title="Return rate", format=".1%"),
                    alt.Tooltip("Orders:Q", format=","),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(chart_h, width="stretch", alt="Average order value by discount rate")

with r3_c3:
    with st.container(border=True):
        st.subheader("I. Return rate by delivery time (%)")
        dd_df = df.assign(
            Bucket=pd.cut(
                df["DeliveryDays"],
                bins=[0, 4, 7, 100],
                labels=["2-4 days", "5-7 days", "8+ days"],
            )
        )
        dd = (
            dd_df.groupby("Bucket", observed=True, as_index=False)
            .agg(RatePct=("Returned", lambda s: s.mean() * 100), Orders=("OrderID", "size"))
        )
        dd["Bucket"] = dd["Bucket"].astype(str)
        bars_i = (
            alt.Chart(dd)
            .mark_bar(color="#c44e52")
            .encode(
                x=alt.X("Bucket:N", sort=["2-4 days", "5-7 days", "8+ days"], title=None),
                y=alt.Y("RatePct:Q", title="% returned"),
                tooltip=[
                    alt.Tooltip("Bucket:N", title="Delivery time"),
                    alt.Tooltip("RatePct:Q", title="Return rate (%)", format=".1f"),
                    alt.Tooltip("Orders:Q", format=","),
                ],
            )
        )
        rule_i = (
            alt.Chart(pd.DataFrame({"y": [overall_rr]}))
            .mark_rule(strokeDash=[5, 5], color="gray")
            .encode(y="y:Q")
        )
        st.altair_chart(
            (bars_i + rule_i).properties(height=260),
            width="stretch",
            alt="Return rate by delivery time",
        )

# ---- Filtered data table ----
with st.expander("Explore filtered orders", expanded=False):
    display_cols = [
        "OrderID",
        "OrderDate",
        "DeliveryDate",
        "DeliveryDays",
        "Region",
        "StoreLocation",
        "Product",
        "CustomerType",
        "Salesperson",
        "Quantity",
        "UnitPrice",
        "DiscountPct",
        "TotalPrice",
        "PaymentMethod",
        "Promotion",
        "Returned",
    ]
    st.dataframe(
        df[display_cols].sort_values("OrderDate", ascending=False),
        width="stretch",
        hide_index=True,
        alt="Filtered product sales orders table",
    )

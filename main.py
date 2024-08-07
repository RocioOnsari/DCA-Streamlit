import dataclasses
import math
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, List
from datetime import datetime, date, timedelta, time
import numpy as np
import scipy.optimize
import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.optimize import curve_fit, fsolve, Bounds

st.set_page_config(layout="wide")


# Function definition
def func(t, a, b):
    return b * a ** t


def func_hyp(t, n, dnh1, qhi):
    return qhi * (1 + n * dnh1 * t) ** (-1 / n)


def func_harm(t, dnh1, qhi):
    return qhi * (1 + dnh1 * t) ** (-1)


# We use curve-fit to calculate an exponential curve that fits the data and returns an array of values that describes the curve
@dataclass
class Coefficents:
    dn: float
    qi: float
    b: float = 1


def coeff_exp(x, y) -> Coefficents:
    popt, pcov = curve_fit(func, x.values, y.values)
    # popt[0] is equal to e^(-Dn)
    etothenegdn = popt[0]
    # We calculate the nominal declination
    dn = math.log(etothenegdn)
    dn = -1 * dn
    return Coefficents(dn=dn, qi=popt[1])


def coeff_hyp(x, y) -> Coefficents:
    # 0 < b < 1
    delta = 0.01
    bounds = ([0 + delta, -np.inf, -np.inf], [1 - delta, np.inf, np.inf])
    popt, pcov = curve_fit(func_hyp, xdata=x.values, ydata=y.values, bounds=bounds)
    print(popt)
    popt[0] = popt[0]
    return Coefficents(dn=popt[1], qi=popt[2], b=popt[0])


def coeff_hyp_ln(x, y) -> Coefficents:
    popt, pcov = curve_fit(f=func_hyp, xdata=x.values, ydata=y.values, bounds=[])
    print(popt)
    popt[0] = popt[0]
    return Coefficents(dn=popt[1], qi=popt[2], b=popt[0])


def coeff_harm(x, y) -> Coefficents:
    popt, pcov = curve_fit(f=func_harm, xdata=x.values, ydata=y.values)
    print(popt)
    return Coefficents(dn=popt[0], qi=popt[1], b=1)


# We draw a semilog date vs rate graph
def graph_semilog_q_date(x_df, y_df):
    hist = px.scatter(df, x_df, y_df, log_y=True, trendline="ols", trendline_options=dict(log_y=True))
    st.plotly_chart(hist)
    return hist


# We draw a semilog date vs rate graph
def graph_semilog_multi(_df, x, ys):
    fig = px.line(_df, x=x, y=ys, log_y=True)
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_layout(hovermode="x unified")
    return fig


# We define a function with the Exponential DCA rate calculation: q=qi*e^(-Dn*DeltaT)
def dca_exp(t: int, coeff: Coefficents) -> float:
    q = coeff.qi * np.exp(-coeff.dn * t)
    if q is None:
        raise RuntimeError(f"Illegal value: {coeff}")
    # st.write(f"{qi} * np.exp(-{c} * {t}) = {q}")
    return q


def dca_hyp(t: int, coeff: Coefficents) -> float:
    return coeff.qi / (np.abs((1 + coeff.b * coeff.dn * t)) ** (1 / coeff.b))


def dca_harm(t: int, coeff: Coefficents) -> float:
    return coeff.qi / (1 + coeff.dn * t)


def denormalize_coef(coefficient: Coefficents, _max_t: float, _max_q: float) -> Coefficents:
    return Coefficents(dn=coefficient.dn / _max_t, qi=coefficient.qi * _max_q, b=coefficient.b)


def reserves_calculation_exp(q, qi, dn):
    return ((q - qi) / dn) / 1000


def reserves_calculation_hyp(q, qi, coeff: Coefficents, dn_norm, offset: float = 0):
    nume = (q * (qi ** coeff.b) - (q ** coeff.b) * qi)
    # st.write(f"({q} * ({qi} ** {coeff.b}) - ({q} ** {coeff.b}) * {qi}) = {nume}")
    deno = ((qi - coeff.b) * dn_norm * (qi ** coeff.b))
    # st.write(f"(({qi} - {coeff.b}) * {dn_norm} * ({qi} ** {coeff.b})) = {deno}")
    final = (nume / deno) / 1000
    return final + offset


def reserves_calculation_arm(q, qi, dn_norm):
    # st.write(f"({q} * np.log({q} / {qi})) / {dn_norm} / 1000 = {dn_norm}")
    return (q * np.log(q / qi)) / dn_norm / 1000


def coeff_table(labels: list[str], coefficients: list[Coefficents]):
    dreservas = {'Dni': map(lambda c: c.dn, coefficients),
                 'Qi': map(lambda c: c.qi, coefficients),
                 'b': map(lambda c: c.b, coefficients)}
    return pd.DataFrame(index=labels, data=dreservas)


def combine_forecasts(base_df, extra_columns: list):
    base = base_df[["Delta", q_column]]
    base = base.set_index("Delta")
    combined = pd.concat([base] + extra_columns, axis=1)
    print(combined.info())
    print(extra_columns)
    combined[date_column] = pozo[date_column].min() + timedelta(
        days=1) * combined.index
    return combined


def forecast(label: str, t_start: int, q_start: float, coeff: Coefficents,
             dca_func: Callable[[int, Coefficents], float],
             q_limit: float = 0.01, n_limit: int = 100, interval_days: int = 30):
    q = q_start
    qi = None
    n = 0
    q_list = []
    t_list = []
    print(f"Starting forecast with{q}")

    while q >= q_limit and n < n_limit:
        t = t_start + n * interval_days
        q = dca_func(t, coeff)
        if qi is None:
            qi = q
        q = (q_start / qi) * q
        q_list.append(q)
        t_list.append(t)
        n = n + 1

    if len(q_list) == 0:
        raise RuntimeError("Bad list lenght")
    return pd.DataFrame(index=t_list, data={label: q_list})


# Graphical user interface
st.title("DCA")

# First, we create a select box in which we will be able to select different datasets
st.text("The units of the dataframe should be m3/d for Q and date should be in MM/DD/AAAA format")
option = st.selectbox(
    'Which production history would you like to choose?',
    ('Pozo A', 'Upload csv', 'Upload excel'))

date_column = "Fecha"
q_column = "Qo [m3/d]"

st.write('You selected:', option)
if option == "Pozo A":
    df = pd.read_csv("data/PozoA.csv", sep=",")
    df = df.drop([12])
    df[date_column] = pd.to_datetime(df[date_column], format="%d/%m/%y", origin="unix")

elif option == "Upload csv":
    st.text("The units of the dataframe should be m3/d for Q and MM/DD/AAAA dor date format")
    # Button to select a separator
    separator = st.text_input("Write the separator used in your file:")
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        # read csv
        df = pd.read_csv(uploaded_file, sep=separator)
        st.write(df)
        date_column = st.selectbox('Select date column', df.columns)
        q_column = st.selectbox('Select w column', df.columns)
        st.write('You selected as date column:', date_column)
        # https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
        format_date = st.radio('Choose the date format:',
                               ["%y/%m/%d", "%d/%m/%y", '%m/%d/%y', "%y-%m-%d", "%d-%m-%y", '%m-%d-%y'])

        df.dropna(inplace=True)

        st.write('You selected as q column:', q_column)

elif option == "Upload excel":
    st.text("The units of the dataframe should be m3/d for Q and MM/DD/AAAA dor date format")
    # Button to select a separator
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # read xlsx
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox('Select the sheet name', options=excel_file.sheet_names)
        skip_rows: int = st.number_input("Write the row number where the data begins", value=0)
        read_rows: int = st.number_input("Write the row number where the data ends", value=100)
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=skip_rows, nrows=read_rows)

        month_or_date = st.checkbox("Use a month column instead of date")
        q_column = st.selectbox('Select q column', df.columns)
        if month_or_date:
            month_column = st.selectbox('Select month column', df.columns)
            start_date = datetime.combine(st.date_input("choose date start"), time(0, 0))
            date_column = month_column + "_DATE"
            df[date_column] = df[month_column].apply(lambda m: start_date + timedelta(days=30 * m))
            st.write(f'{date_column} built from {min(df[month_column])} to {max(df[month_column])}')
        else:
            date_column = st.selectbox('Select date column', df.columns)

        st.write(f'You selected as date column: {date_column} from {min(df[date_column])} to {max(df[date_column])}')
        st.write('You selected as q column:', q_column)

else:
    st.stop()
with st.expander("Input Table", False):
    st.table(df)

pozo = df
print(pozo)
print(pozo.info())
pozo["Delta"] = (pozo[date_column].diff() / timedelta(days=1)).cumsum()
pozo['Delta'].iat[0] = 0
interval_days_param = int(pozo['Delta'].iat[1])

max_t = pozo['Delta'].max()
max_q = pozo[q_column].max()
q0 = pozo[q_column][0]
pozo['Delta Norm'] = pozo['Delta'] / max_t
pozo["Qo norm"] = pozo[q_column] / max_q
delta_t_norm = pozo['Delta Norm']
qhist = pozo[q_column]
qhist_norm = pozo["Qo norm"]

st.table(pd.DataFrame(index=["Value"], data={
    'Q0': [q0],
    'max_t': [max_t],
    'max_q': [max_q],
    'delta_days': [interval_days_param]
}))
st.write(f"Q0: {q0}, max_t: {max_t}, max_q: {max_q}")
with st.expander("Mod Input Table", False):
    st.table(pozo)

# Now we draw the graph with the data
st.header("Qo [m3/d] vs Date graph")
hist_graph = graph_semilog_q_date(pozo[date_column], pozo[q_column])

ini, fin = st.select_slider(label="Elegir rango para reducir", value=(0, len(delta_t_norm) - 1),
                            options=list(range(len(delta_t_norm))))

exp_coeff = denormalize_coef(coeff_exp(delta_t_norm[ini:fin], qhist_norm[ini:fin]), _max_t=max_t, _max_q=max_q)
hyper_coeff = denormalize_coef(coeff_hyp(delta_t_norm[ini:fin], qhist_norm[ini:fin]), _max_t=max_t, _max_q=max_q)
harmonic_coeff = denormalize_coef(coeff_harm(delta_t_norm[ini:fin], qhist_norm[ini:fin]), _max_t=max_t, _max_q=max_q)

st.table(coeff_table(["1P", "2P", "3P"], [exp_coeff, hyper_coeff, harmonic_coeff]))

# Here, we select the economic limit of the well
ec_limit = st.number_input("Select the minimum Q [m3/d] accepted:", value=1)

# Last time value
final_history_date = pozo["Delta"].values[-1]

# Last historic rate value
final_history_rate = pozo[q_column].values[-1]

# Graphical user interface
st.text(f' Final historic rate: {final_history_rate}')

# Run the 3 forecasts, for each declination type
qexp_forecast = forecast("Qexp", t_start=final_history_date, q_start=final_history_rate, coeff=exp_coeff,
                         dca_func=dca_exp)
q2p_forecast = forecast("Q2p", t_start=final_history_date, q_start=final_history_rate, coeff=hyper_coeff,
                        dca_func=dca_hyp)
qharm_forecast = forecast("Qharm", t_start=final_history_date, q_start=final_history_rate, coeff=harmonic_coeff,
                          dca_func=dca_harm)

# We calculate reserves
P90 = reserves_calculation_exp(final_history_rate, ec_limit, exp_coeff.dn)
dn_norm = hyper_coeff.dn * ((final_history_rate / hyper_coeff.qi) ** hyper_coeff.b)
P50 = reserves_calculation_hyp(final_history_rate, ec_limit, hyper_coeff, dn_norm)

hcoeff = dataclasses.replace(hyper_coeff)
hcoeff.b = 1
P10 = reserves_calculation_arm(final_history_rate, ec_limit, dn_norm)
P10_LN = (P50 ** 2) / P90

b = \
    fsolve(
        func=lambda b: reserves_calculation_hyp(final_history_rate, 1, Coefficents(qi=0, dn=0, b=b), dn_norm, -P10_LN),
        x0=0.3)[0]
hyper_3p_coeff = hyper_coeff
hyper_3p_coeff.b = b
qharm_3p_forecast = forecast("Qharm 3P", t_start=final_history_date, q_start=final_history_rate, coeff=hyper_3p_coeff,
                             dca_func=dca_hyp)

# Combine the forecasts with the data
decline_forecasts_df = combine_forecasts(pozo, [qexp_forecast, q2p_forecast, qharm_forecast, qharm_3p_forecast])
# Plot and show table
with st.expander("Complete Table", False):
    st.table(decline_forecasts_df)
st.plotly_chart(graph_semilog_multi(decline_forecasts_df, date_column, [q_column, "Qexp", "Q2p", "Qharm", "Qharm 3P"]))

dreservas = {'P10': P10, 'P10_LN': P10_LN, 'P50': P50, 'P90': P90}
st.subheader("Reservas")
dfr = pd.DataFrame(index=["Reservas"], data=dreservas)
st.table(dfr)

st.subheader("Ahora calculamos los coeficientes con distintos puntos")
start, end = st.select_slider(label="Elegir rango", value=(0, len(delta_t_norm) - 1),
                              options=list(range(len(delta_t_norm))))


@dataclass
class Method:
    name: str
    coefficient_func: Callable
    dca_func: Callable


@dataclass
class MethodResult:
    name: str
    normalized_coefficient: Coefficents
    denormalized_coefficient: Coefficents
    forecast_df: pd.DataFrame


exp_method = Method("exp", coefficient_func=coeff_exp, dca_func=dca_exp)
hyp_method = Method("hyp", coefficient_func=coeff_hyp, dca_func=dca_hyp)
harm_method = Method("harm", coefficient_func=coeff_harm, dca_func=dca_harm)
methods = [exp_method, hyp_method, harm_method]

selected_methods = st.multiselect("Select methods to use", options=methods, format_func=lambda x: x.name)


def calculate_method_result(experiment_name: str, method: Method, ts: List[int], ys: list[float],
                            _max_t: float, _max_q: float, _t_start: int, _q_start: float) -> MethodResult:
    norm_coef = method.coefficient_func(ts, ys)
    name = f"{experiment_name} {method.name}"
    denorm_coef = denormalize_coef(norm_coef, _max_t=_max_t, _max_q=_max_q)
    forecast_df = forecast(name, t_start=_t_start, q_start=_q_start, coeff=denorm_coef,
                           dca_func=method.dca_func)
    return MethodResult(name=name, normalized_coefficient=norm_coef,
                        denormalized_coefficient=denorm_coef,
                        forecast_df=forecast_df)


results: List[MethodResult] = [calculate_method_result(f"All", method, delta_t_norm, qhist_norm, _max_t=max_t,
                                                       _max_q=max_q, _t_start=final_history_date, _q_start=final_history_rate)
                               for method in selected_methods] + [
                                  calculate_method_result(f"{start}:{end}", method, delta_t_norm[start:end],
                                                          qhist_norm[start:end], _max_t=max_t, _max_q=max_q, _t_start=delta_t_norm[end]*max_t, _q_start=qhist_norm[end]*max_q)
                                  for method in selected_methods]

# Armar la tabla de coeficientes
points_coeffs_df = coeff_table([result.name for result in results],
                               [result.denormalized_coefficient for result in results])
# Calcular las reservas
points_coeffs_df["P10"] = points_coeffs_df["Dni"].apply(
    lambda dn: reserves_calculation_exp(final_history_rate, ec_limit, dn))
st.table(points_coeffs_df)

custom_func = dca_hyp

point_forecasts_df = combine_forecasts(pozo, [result.forecast_df for result in results])
with st.expander("Point Forecasts Table", False):
    st.table(point_forecasts_df)

st.plotly_chart(graph_semilog_multi(point_forecasts_df, date_column, [q_column] + [result.name for result in results]))


import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# ---------------------- Page Setup ----------------------
st.set_page_config(
    page_title="PIF · Optimización Gestión Editorial",
    page_icon="📈",
    layout="wide"
)

st.title("📈 PIF · Modelado y optimización de la productividad en la gestión editorial")
st.caption("Función base:  P(x,y) = x · y · exp(-αx) · exp(-βy)")

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.header("⚙️ Parámetros del modelo")
    col_a, col_b = st.columns(2)
    with col_a:
        alpha = st.number_input("α (penalización x)", min_value=0.0, max_value=1.0, value=0.10, step=0.01, format="%.2f")
    with col_b:
        beta = st.number_input("β (penalización y)", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f")

    st.markdown("---")
    st.subheader("Rangos de simulación")
    max_x = st.slider("Máximo de redactores (x)", 5, 200, 60, step=5)
    max_y = st.slider("Máximo de artículos (y)", 5, 300, 100, step=5)
    step_xy = st.slider("Resolución de la malla", 20, 200, 80, step=10)

    st.markdown("---")
    st.subheader("Puntos de interés")
    show_theoretical = st.checkbox("Mostrar óptimo analítico (x*, y*) = (1/α, 1/β)", value=True)
    custom_point = st.checkbox("Comparar un punto manual", value=False)
    x_manual = y_manual = None
    if custom_point:
        col1, col2 = st.columns(2)
        with col1:
            x_manual = st.number_input("x (redactores)", min_value=0.0, max_value=10000.0, value=10.0, step=1.0)
        with col2:
            y_manual = st.number_input("y (artículos)", min_value=0.0, max_value=10000.0, value=20.0, step=1.0)

    st.markdown("---")
    st.subheader("Descarga de datos")
    want_csv = st.checkbox("Generar CSV de la superficie", value=False)

# ---------------------- Helpers ----------------------
def P(x, y, a, b):
    return x * y * np.exp(-a * x) * np.exp(-b * y)

# ---------------------- Compute Grid ----------------------
x_vals = np.linspace(0, max_x, step_xy)
y_vals = np.linspace(0, max_y, step_xy)
X, Y = np.meshgrid(x_vals, y_vals, indexing="xy")
Z = P(X, Y, alpha, beta)

# Theoretical optimum (interior critical point) if alpha, beta > 0
x_star = 1.0 / alpha if alpha > 0 else None
y_star = 1.0 / beta if beta > 0 else None
p_star = P(x_star, y_star, alpha, beta) if (x_star is not None and y_star is not None) else None

# Grid maximum (on the simulated mesh)
flat_idx = int(np.argmax(Z))
x_mesh_opt = float(X.flatten()[flat_idx])
y_mesh_opt = float(Y.flatten()[flat_idx])
p_mesh_opt = float(Z.flatten()[flat_idx])

# ---------------------- Header KPIs ----------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("α (x)", f"{alpha:.2f}")
with kpi2:
    st.metric("β (y)", f"{beta:.2f}")
with kpi3:
    st.metric("P* teórica", f"{p_star:,.3f}" if p_star is not None else "—")
with kpi4:
    st.metric("Óptimo (x*, y*)", f"({x_star:.2f}, {y_star:.2f})" if (x_star is not None and y_star is not None) else "—")

st.markdown("")
st.markdown("### 🔍 Interpretación rápida")
st.write(
    "- **x**: número de redactores.  \n"
    "- **y**: número de artículos en curso.  \n"
    "- **α, β**: penalizaciones por saturación/ineficiencia al crecer x o y.  \n"
    "- El óptimo analítico interno (si existe) es **(x*, y*) = (1/α, 1/β)**."
)

# ---------------------- Plots ----------------------
tab3d, tab2d, tabsheet = st.tabs([
    "🌋 Superficie 3D", 
    "🗺️ Contorno (curvas de nivel)", 
    "📄 Tabla de datos"
])

with tab3d:
    fig = go.Figure(
        data=[
            go.Surface(
                z=Z,
                x=X,
                y=Y,
                colorbar=dict(title="P(x,y)"),
                opacity=0.95
            )
        ]
    )
    fig.update_layout(
        title="Superficie de productividad P(x,y)",
        scene=dict(
            xaxis_title="x (redactores)",
            yaxis_title="y (artículos)",
            zaxis_title="P(x,y)"
        ),
        height=650
    )

    # Mark theoretical optimum if within plotting range
    if show_theoretical and (x_star is not None and y_star is not None):
        if 0 <= x_star <= max_x and 0 <= y_star <= max_y:
            z_star = P(x_star, y_star, alpha, beta)
            fig.add_trace(
                go.Scatter3d(
                    x=[x_star], y=[y_star], z=[z_star],
                    mode="markers+text",
                    text=[f"Óptimo ({x_star:.2f}, {y_star:.2f})"],
                    textposition="top center",
                    marker=dict(size=6, symbol="diamond")
                )
            )

    # Manual point comparison
    if custom_point and x_manual is not None and y_manual is not None:
        if 0 <= x_manual <= max_x and 0 <= y_manual <= max_y:
            z_manual = P(x_manual, y_manual, alpha, beta)
            fig.add_trace(
                go.Scatter3d(
                    x=[x_manual], y=[y_manual], z=[z_manual],
                    mode="markers+text",
                    text=[f"P({x_manual:.0f}, {y_manual:.0f})={z_manual:,.2f}"],
                    textposition="top center",
                    marker=dict(size=5)
                )
            )

    # Also mark the grid maximum found on the mesh
    if 0 <= x_mesh_opt <= max_x and 0 <= y_mesh_opt <= max_y:
        fig.add_trace(
            go.Scatter3d(
                x=[x_mesh_opt], y=[y_mesh_opt], z=[p_mesh_opt],
                mode="markers+text",
                text=[f"Máximo malla ({x_mesh_opt:.2f}, {y_mesh_opt:.2f})"],
                textposition="bottom center",
                marker=dict(size=6, symbol="x")
            )
        )

    st.plotly_chart(fig, use_container_width=True)

with tab2d:
    fig2 = go.Figure(
        data=go.Contour(
            z=Z,
            x=x_vals,
            y=y_vals,
            contours=dict(showlabels=True),
            colorbar=dict(title="P(x,y)")
        )
    )
    fig2.update_layout(
        title="Curvas de nivel de P(x,y)",
        xaxis_title="x (redactores)",
        yaxis_title="y (artículos)",
        height=650
    )

    # Overlay optimum
    if show_theoretical and (x_star is not None and y_star is not None):
        if 0 <= x_star <= max_x and 0 <= y_star <= max_y:
            fig2.add_trace(
                go.Scatter(
                    x=[x_star], y=[y_star],
                    mode="markers+text",
                    text=[f"Óptimo ({x_star:.2f}, {y_star:.2f})"],
                    textposition="top center"
                )
            )

    # Overlay manual point
    if custom_point and x_manual is not None and y_manual is not None:
        if 0 <= x_manual <= max_x and 0 <= y_manual <= max_y:
            fig2.add_trace(
                go.Scatter(
                    x=[x_manual], y=[y_manual],
                    mode="markers+text",
                    text=[f"P={P(x_manual, y_manual, alpha, beta):,.2f}"],
                    textposition="bottom right"
                )
            )

    # Mark the grid maximum on the contour as well
    fig2.add_trace(
        go.Scatter(
            x=[x_mesh_opt], y=[y_mesh_opt],
            mode="markers+text",
            text=[f"Máx malla ({x_mesh_opt:.2f}, {y_mesh_opt:.2f})"],
            textposition="bottom center"
        )
    )

    st.plotly_chart(fig2, use_container_width=True)

with tabsheet:
    df = pd.DataFrame({
        "x": X.flatten(),
        "y": Y.flatten(),
        "P(x,y)": Z.flatten()
    })
    # Controls for table view
    view_mode = st.radio("Vista de tabla", ["Top por productividad", "Malla completa"], horizontal=True)
    if view_mode == "Top por productividad":
        top_n = st.slider("Mostrar top N (por P)", 10, 1000, 200, step=10)
        df_sorted = df.sort_values("P(x,y)", ascending=False).head(top_n)
        st.dataframe(df_sorted.reset_index(drop=True))
        st.caption(f"Mostrando los {top_n} puntos con mayor productividad en la malla simulada.")
        # show the single best point on the mesh
        st.markdown("**Máximo en la malla**")
        st.table(pd.DataFrame([{
            "x* (malla)": x_mesh_opt,
            "y* (malla)": y_mesh_opt,
            "P* (malla)": p_mesh_opt
        }]))
    else:
        st.dataframe(df)  # full grid (may be large)

    if want_csv:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar CSV (malla completa)", data=csv, file_name="superficie_productividad.csv", mime="text/csv")

# ---------------------- Explainers ----------------------
with st.expander("🧠 ¿Cómo interpretar el modelo?"):
    st.markdown(
        "- Crecimientos con rendimientos decrecientes: x e y aumentan la productividad al inicio, "
        "pero la penalización exponencial captura la saturación (más gente y más tareas reducen eficiencia).\n"
        "- Óptimo analítico: si α, β > 0, el máximo interior ocurre en x* = 1/α, y* = 1/β. "
        "Por ejemplo, con α=0.10 y β=0.05, el óptimo es (10, 20).\n"
        "- Uso práctico: mueve α y β para reflejar realidades de tu equipo (mayor β si muchos artículos en paralelo "
            "saturan rápido; mayor α si coordinar más redactores se vuelve costoso)."
    )

with st.expander("🧪 Sugerencias de experimentación"):
    st.markdown(
        "- Simula límites: reduce el rango máximo de x o y para representar presupuestos o calendarios restringidos.\n"
        "- Comparación de escenarios: activa “Comparar un punto manual” y contrasta con el óptimo analítico.\n"
        "- Si quieres, podemos normalizar P a un índice 0-100 para leerlo como eficiencia porcentual."
    )

st.success("Tabla actualizada: ahora puedes ver el Top por productividad y el máximo hallado en la malla.")

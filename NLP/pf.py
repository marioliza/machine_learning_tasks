import streamlit as st

# Estilos personalizados
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .header {
        font-size: 48px;
        font-weight: bold;
        color: #1E3D58; /* Azul oscuro */
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 24px;
        color: #1E3D58; /* Azul oscuro */
        margin-top: 20px;
    }
    .project-section {
        background-color: #ffffff; /* Blanco */
        padding: 20px;
        margin-top: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-left: 5px solid #1E3D58; /* Azul oscuro */
    }
    .contact {
        margin-top: 50px;
        font-size: 18px;
        color: #555555; /* Gris */
        text-align: center;
    }
    .project-title {
        color: #1E3D58; /* Azul oscuro */
        font-weight: bold;
        font-size: 22px;
    }
    .project-description {
        color: #555555; /* Gris */
    }
    .link {
        color: #1E3D58; /* Azul oscuro */
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Título de la aplicación
st.markdown('<div class="header">Portafolio de Proyectos en Ciencias de Datos</div>', unsafe_allow_html=True)

# Descripción introductoria
st.write("""
Bienvenido a mi portafolio de proyectos en ciencias de datos. Aquí encontrarás ejemplos de trabajos que he realizado utilizando diferentes técnicas y herramientas de análisis de datos.
""")

# Sección de proyectos
st.markdown('<div class="subheader">Proyectos</div>', unsafe_allow_html=True)

# Proyecto 1: Modelo de recomendación multicapa
#st.markdown('<div class="project-section">', unsafe_allow_html=True)
st.markdown('<div class="project-title">Modelo de Recomendación Multicapa</div>', unsafe_allow_html=True)
st.write("""
<div class="project-description">
Este proyecto consiste en un modelo de recomendación multicapa desarrollado utilizando PyTorch. El objetivo del modelo es recomendar productos a los usuarios basándose en sus preferencias anteriores.
</div>
""", unsafe_allow_html=True)
#st.image("https://example.com/path_to_your_image.jpg", caption="Imagen del modelo de recomendación")
st.markdown('<a href="https://github.com/tu_usuario/recomendacion_multicapa" class="link">Ver código en GitHub</a>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Proyecto 2: Modelo de Forecasting con ARIMA
st.markdown('<div class="project-section">', unsafe_allow_html=True)
st.markdown('<div class="project-title">Modelo de Forecasting con ARIMA</div>', unsafe_allow_html=True)
st.write("""
<div class="project-description">
En este proyecto, utilicé el modelo ARIMA para realizar predicciones de series temporales. El modelo fue evaluado utilizando métricas estándar y comparado con otros enfoques de forecasting.
</div>
""", unsafe_allow_html=True)
#st.image("https://example.com/path_to_your_image2.jpg", caption="Imagen del modelo ARIMA")
st.markdown('<a href="https://github.com/tu_usuario/forecasting_arima" class="link">Ver código en GitHub</a>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Puedes añadir más proyectos siguiendo la misma estructura

# Pie de página
st.markdown('<div class="contact">', unsafe_allow_html=True)
st.write("### Contacto")
st.write("Puedes contactarme en [tuemail@dominio.com](mailto:tuemail@dominio.com)")
st.markdown('</div>', unsafe_allow_html=True)

# app.py
import os
import requests
# Get API URL from environment variable with fallback
API_URL = os.getenv('API_URL', 'http://localhost:8000')
import plotly.graph_objects as go
from PIL import Image
import io
import streamlit as st

# Configure the page
st.set_page_config(
    page_title="Bias Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .big-font {
        font-size: 36px !important;
        color: #FF5733;
    }
    .header-font {
        font-size: 48px !important;
        color: #3498DB;
        font-weight: bold;
    }
    .subheader-font {
        font-size: 30px !important;
        color: #2ECC71;
    }
    </style>
""", unsafe_allow_html=True)

def create_gauge_chart(probability, title):
    """
    Create a gauge chart for probability visualization
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=200)
    return fig

def create_horizontal_bar_chart(probabilities, title):
    """
    Create a horizontal bar chart for probability visualization
    """
    fig = go.Figure(go.Bar(
        x=list(probabilities.values()),
        y=list(probabilities.keys()),
        orientation='h',
        marker_color='rgb(26, 118, 255)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Probability',
        yaxis_title='Class',
        yaxis={'categoryorder':'total ascending'},
        height=400
    )
    return fig

# Main app
st.markdown('<h1 class="header-font">Bias Detection Dashboard</h1>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
    
    # Make prediction
    files = {'file': ('image.jpg', uploaded_file, 'image/jpeg')}
    response = requests.post('http://localhost:8000/predict/', files=files)
    
    if response.status_code == 200:
        predictions = response.json()
        
        with col2:
            # Gender predictions
            st.markdown('<h2 class="subheader-font">Gender Classification</h2>', unsafe_allow_html=True)
            gender_probs = predictions['gender_probabilities']
            max_gender = max(gender_probs.items(), key=lambda x: x[1])
            st.markdown(f"<p class='big-font'>Predicted Gender: {max_gender[0]} ({max_gender[1]:.2%})</p>", unsafe_allow_html=True)
            
            # Display gender probabilities as gauge charts
            for gender, prob in gender_probs.items():
                st.plotly_chart(create_gauge_chart(prob, f"{gender} Probability"))
            
            # Race predictions
            st.markdown('<h2 class="subheader-font">Race Classification</h2>', unsafe_allow_html=True)
            race_probs = predictions['race_probabilities']
            max_race = max(race_probs.items(), key=lambda x: x[1])
            st.markdown(f"<p class='big-font'>Predicted Race: {max_race[0]} ({max_race[1]:.2%})</p>", unsafe_allow_html=True)
            
            # Display race probabilities as a horizontal bar chart
            st.plotly_chart(create_horizontal_bar_chart(race_probs, "Race Probabilities"))
    else:
        st.error("Error making prediction. Please try again.")

# Add some instructions in the sidebar
with st.sidebar:
    st.header("Instructions")
    st.write("""
    1. Upload an image using the file uploader
    2. The system will automatically process the image
    3. View the prediction results in the dashboard
    4. The gauge charts show gender probabilities
    5. The horizontal bar chart shows race probabilities
    """)
    
    st.header("About")
    st.write("""
    This dashboard uses two separate deep learning models:
    - A binary classifier for gender prediction
    - A multi-class classifier for race prediction
    
    The models run in parallel for optimal performance.
    """)
import os
import requests
import plotly.graph_objects as go
from PIL import Image
import io
import streamlit as st

# Get API URL from environment variable with fallback
API_URL = os.getenv('API_URL', 'https://face-gender-pred.onrender.com')

# Configure the page with a dark theme
st.set_page_config(
    page_title="Bias Detection Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/yourusername/yourrepo',
        'Report a bug': "https://github.com/yourusername/yourrepo/issues",
        'About': "# Bias Detection Dashboard\nThis application detects potential bias in facial recognition systems."
    }
)

# Custom CSS for styling
st.markdown("""
    <style>
    .big-font {
        font-size: 36px !important;
        color: #FF5733;
        margin-bottom: 20px;
    }
    .header-font {
        font-size: 48px !important;
        color: #3498DB;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        margin-bottom: 30px;
    }
    .subheader-font {
        font-size: 30px !important;
        color: #2ECC71;
        padding: 10px;
        margin-bottom: 20px;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }
    div[data-testid="stImage"] > img {
        width: 600px !important;
        max-width: 100%;
        margin: auto;
        display: block;
    }
    </style>
""", unsafe_allow_html=True)

def create_gauge_chart(probability, title):
    """Create a gauge chart for probability visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
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
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        font={'size': 16}
    )
    return fig

def create_horizontal_bar_chart(probabilities, title):
    """Create a horizontal bar chart for probability visualization"""
    fig = go.Figure(go.Bar(
        x=list(probabilities.values()),
        y=list(probabilities.keys()),
        orientation='h',
        marker_color='rgb(26, 118, 255)',
        text=[f'{v:.1%}' for v in probabilities.values()],
        textposition='auto',
    ))
    
    fig.update_layout(
        title={'text': title, 'font': {'size': 24}},
        xaxis_title='Probability',
        yaxis_title='Class',
        yaxis={'categoryorder':'total ascending'},
        height=600,  # Increased height for the race chart
        margin=dict(l=20, r=20, t=40, b=20),
        font={'size': 16}
    )
    return fig

# Main app
st.markdown('<h1 class="header-font">Bias Detection Dashboard</h1>', unsafe_allow_html=True)

# File uploader with better error handling
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image')
        
        with st.spinner('Processing image...'):  # Add loading spinner
            # Make prediction with timeout and error handling
            files = {'file': ('image.jpg', uploaded_file, 'image/jpeg')}
            try:
                response = requests.post(
                    f'{API_URL}/predict/',
                    files=files,
                    timeout=30  # Add timeout
                )
                
                if response.status_code == 200:
                    predictions = response.json()
                    
                    # Gender Classification Section
                    st.markdown('<h2 class="subheader-font">Gender Classification</h2>', unsafe_allow_html=True)
                    gender_probs = predictions['gender_probabilities']
                    max_gender = max(gender_probs.items(), key=lambda x: x[1])
                    st.markdown(f"<p class='big-font'>Predicted Gender: {max_gender[0]} ({max_gender[1]:.2%})</p>", unsafe_allow_html=True)
                    
                    # Create two columns for gender gauge charts
                    gauge_col1, gauge_col2 = st.columns(2)
                    
                    # Display gender probabilities side by side
                    for (gender, prob), col in zip(gender_probs.items(), [gauge_col1, gauge_col2]):
                        with col:
                            st.plotly_chart(create_gauge_chart(prob, f"{gender} Probability"), use_container_width=True)
                    
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    
                    # Race Classification Section
                    st.markdown('<h2 class="subheader-font">Race Classification</h2>', unsafe_allow_html=True)
                    race_probs = predictions['race_probabilities']
                    max_race = max(race_probs.items(), key=lambda x: x[1])
                    st.markdown(f"<p class='big-font'>Predicted Race: {max_race[0]} ({max_race[1]:.2%})</p>", unsafe_allow_html=True)
                    
                    st.plotly_chart(create_horizontal_bar_chart(race_probs, "Race Probabilities"), use_container_width=True)
                    
                elif response.status_code == 503:
                    st.error("The API service is currently unavailable. Please try again later.")
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
                    st.error(f"Response: {response.text}")
            
            except requests.exceptions.Timeout:
                st.error("Request timed out. The server took too long to respond.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Please check if the service is available.")
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred while connecting to the API: {str(e)}")
                
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        if st.button('Try Again'):
            st.experimental_rerun()

# Add instructions in the sidebar
with st.sidebar:
    st.header("Instructions")
    st.write("""
    1. Upload an image using the file uploader above
    2. Wait for the automatic processing
    3. View detailed predictions in the dashboard
    4. Gender probabilities are shown in gauge charts
    5. Race probabilities are displayed in a bar chart
    """)
    
    st.header("About")
    st.write("""
    This dashboard uses two deep learning models:
    - A binary classifier for gender prediction
    - A multi-class classifier for race prediction
    
    The models run in parallel for optimal performance.
    """)
    
    # Add version info
    st.sidebar.markdown("---")
    st.sidebar.markdown("v1.0.0")

# Handle server timeouts
if not st.session_state.get('has_timeout_handler'):
    st.session_state['has_timeout_handler'] = True
    if st.session_state.get('timeout_occurred'):
        st.error("Previous request timed out. Please try again.")
        st.session_state['timeout_occurred'] = False
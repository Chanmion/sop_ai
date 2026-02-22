# sop_ai
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git build-essential
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install streamlit     langchain     langchain-community     langchain-huggingface     faiss-cpu     sentence-transformers     transformers     torch     pypdf
streamlit run chat_app.py --server.address 0.0.0.0 --server.port 8501

#\!/bin/bash

echo "FinRAG Setup"
echo "=============="

# Check Python
echo "Checking Python..."
if \! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.10+"
    exit 1
fi
echo "OK: Python $(python3 --version | awk '{print $2}')"

# Create venv
echo ""
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
echo "OK: Virtual environment created"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "OK: Dependencies installed"

# Check PostgreSQL
echo ""
echo "Checking PostgreSQL..."
if \! command -v psql &> /dev/null; then
    echo "WARNING: PostgreSQL not found. Install with:"
    echo "  brew install postgresql"
    echo "  brew services start postgresql"
else
    echo "OK: PostgreSQL found"
    echo ""
    echo "Creating database..."
    createdb finrag 2>/dev/null || echo "Database already exists"
    echo "OK: Database ready"
fi

# Download spacy model
echo ""
echo "Installing spaCy NER model..."
python3 -m spacy download en_core_web_sm -q
echo "OK: spaCy model installed"

echo ""
echo "=============================================="
echo "Setup Complete"
echo ""
echo "Next steps:"
echo "  1. Set environment variables (see README)"
echo "  2. Load data: python load_real_data_simple.py"
echo "  3. Test retrieval: python test_retrieval.py"
echo "  4. Test generation: python test_generation.py"
echo ""

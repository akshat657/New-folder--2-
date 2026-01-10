"""Health check script to verify environment setup"""
import os
import sys
from dotenv import load_dotenv


def check_environment():
    """Verify all required dependencies and configuration"""
    load_dotenv()
    
    print("🏥 Running health checks.. .\n")
    
    issues = []
    warnings = []
    
    # Check Python version
    if sys.version_info < (3, 9):
        issues.append("❌ Python 3.9+ required")
    else:
        print(f"✅ Python {sys.version. split()[0]}")
    
    # Check API keys
    if not os.getenv("GROQ_API_KEY"):
        issues.append("❌ GROQ_API_KEY not found in . env")
    else:
        print("✅ GROQ_API_KEY configured")
    
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key: 
        warnings.append("⚠️  GOOGLE_API_KEY not found (optional)")
    else:
        print("✅ GOOGLE_API_KEY configured")
    
    # Check critical imports
    required_packages = [
        ("streamlit", "Streamlit"),
        ("langchain", "LangChain"),
        ("langchain_groq", "LangChain Groq"),
        ("langchain_huggingface", "LangChain HuggingFace"),
        ("faiss", "FAISS"),
        ("PyPDF2", "PyPDF2"),
    ]
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name} installed")
        except ImportError:
            issues.append(f"❌ {name} not installed")
    
    # Summary
    print("\n" + "="*50)
    
    if issues:
        print("\n❌ CRITICAL ISSUES:")
        for issue in issues: 
            print(f"  {issue}")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings: 
            print(f"  {warning}")
    
    if not issues and not warnings:
        print("\n✅ All checks passed!  You're ready to go!")
        return True
    elif not issues:
        print("\n⚠️  Some warnings, but you can proceed")
        return True
    else:
        print("\n❌ Please fix the issues above before running the app")
        return False


if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
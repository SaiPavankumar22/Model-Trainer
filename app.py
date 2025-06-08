from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
import pandas as pd
import os
import pickle
from ydata_profiling import ProfileReport
from sklearn.utils.multiclass import type_of_target
import numpy as np
from werkzeug.utils import secure_filename

# PyCaret imports
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, save_model as reg_save, load_model as reg_load
from pycaret.classification import setup as clf_setup, compare_models as clf_compare, pull as clf_pull, save_model as clf_save, load_model as clf_load

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure secret key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Global variables to store data and model info
current_df = None
current_model = None
model_type = None
comparison_results = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}

def determine_problem_type(target_series):
    """
    Determine if the problem is regression or classification based on target variable
    """
    target_type = type_of_target(target_series)
    
    # Handle different target types
    if target_type in ['continuous', 'continuous-multioutput']:
        return 'regression'
    elif target_type in ['binary', 'multiclass', 'multiclass-multioutput', 'multilabel-indicator']:
        return 'classification'
    else:
        # For unknown types, make an educated guess
        unique_values = target_series.nunique()
        total_values = len(target_series)
        
        # If unique values are less than 10% of total values or less than 20, treat as classification
        if unique_values < max(total_values * 0.1, 20):
            return 'classification'
        else:
            return 'regression'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global current_df
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                current_df = pd.read_csv(filepath, index_col=None)
                # Save as dataset.csv for consistency
                current_df.to_csv('dataset.csv', index=None)
                flash(f'File uploaded successfully! Dataset shape: {current_df.shape}')
                return redirect(url_for('profiling'))
            except Exception as e:
                flash(f'Error reading file: {str(e)}')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload a CSV file.')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/profiling')
def profiling():
    global current_df
    
    if current_df is None:
        if os.path.exists('dataset.csv'):
            current_df = pd.read_csv('dataset.csv', index_col=None)
        else:
            flash('No dataset found. Please upload a dataset first.')
            return redirect(url_for('upload'))
    
    try:
        # Generate profiling report
        profile = ProfileReport(current_df, title="Dataset Profiling Report", explorative=True)
        report_path = 'reports/profile_report.html'
        profile.to_file(report_path)
        
        # Get basic dataset info
        dataset_info = {
            'shape': current_df.shape,
            'columns': list(current_df.columns),
            'dtypes': current_df.dtypes.to_dict(),
            'missing_values': current_df.isnull().sum().to_dict(),
            'head': current_df.head().to_html(classes='table table-striped')
        }
        
        return render_template('profiling.html', dataset_info=dataset_info, report_exists=True)
    except Exception as e:
        flash(f'Error generating profile: {str(e)}')
        return redirect(url_for('upload'))

@app.route('/view_report')
def view_report():
    report_path = 'reports/profile_report.html'
    if os.path.exists(report_path):
        return send_file(report_path)
    else:
        flash('Profile report not found. Please generate it first.')
        return redirect(url_for('profiling'))

@app.route('/modelling', methods=['GET', 'POST'])
def modelling():
    global current_df, current_model, model_type, comparison_results
    
    if current_df is None:
        if os.path.exists('dataset.csv'):
            current_df = pd.read_csv('dataset.csv', index_col=None)
        else:
            flash('No dataset found. Please upload a dataset first.')
            return redirect(url_for('upload'))
    
    if request.method == 'POST':
        target_column = request.form.get('target_column')
        
        if not target_column or target_column not in current_df.columns:
            flash('Please select a valid target column.')
            return redirect(request.url)
        
        try:
            # Determine problem type
            target_series = current_df[target_column].dropna()
            model_type = determine_problem_type(target_series)
            
            flash(f'Detected problem type: {model_type.upper()}')
            
            # Setup PyCaret based on problem type
            if model_type == 'regression':
                setup_result = reg_setup(current_df, target=target_column, session_id=123, train_size=0.8)
                setup_df = reg_pull()
                
                # Compare models
                best_model = reg_compare(include=['lr', 'rf', 'dt', 'gbr', 'ridge', 'lasso'])
                comparison_results = reg_pull()
                
                # Save the best model
                reg_save(best_model, 'models/best_model')
                current_model = best_model
                
            else:  # classification
                setup_result = clf_setup(current_df, target=target_column, session_id=123, train_size=0.8)
                setup_df = clf_pull()
                
                # Compare models
                best_model = clf_compare(include=['lr', 'rf', 'dt', 'gbc', 'nb', 'svm'])
                comparison_results = clf_pull()
                
                # Save the best model
                clf_save(best_model, 'models/best_model')
                current_model = best_model
            
            # Convert DataFrames to HTML for display
            setup_html = setup_df.to_html(classes='table table-striped table-sm')
            comparison_html = comparison_results.to_html(classes='table table-striped table-sm')
            
            flash(f'Model training completed! Best model type: {type(best_model).__name__}')
            
            return render_template('modelling_results.html', 
                                 setup_html=setup_html,
                                 comparison_html=comparison_html,
                                 model_type=model_type,
                                 target_column=target_column)
            
        except Exception as e:
            flash(f'Error during modeling: {str(e)}')
            return redirect(request.url)
    
    # GET request - show modeling form
    columns = list(current_df.columns)
    return render_template('modelling.html', columns=columns)

@app.route('/download')
def download():
    model_path = 'models/best_model.pkl'
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True, download_name='best_model.pkl')
    else:
        flash('No trained model found. Please train a model first.')
        return redirect(url_for('modelling'))

@app.route('/api/dataset_info')
def api_dataset_info():
    """API endpoint to get dataset information"""
    global current_df
    
    if current_df is None:
        return jsonify({'error': 'No dataset loaded'}), 404
    
    info = {
        'shape': current_df.shape,
        'columns': list(current_df.columns),
        'dtypes': {col: str(dtype) for col, dtype in current_df.dtypes.items()},
        'missing_values': current_df.isnull().sum().to_dict()
    }
    
    return jsonify(info)

# Create HTML templates
def create_templates():
    templates_dir = 'templates'
    os.makedirs(templates_dir, exist_ok=True)
    
    # Base template
    base_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoNickML - {% block title %}{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .navbar-brand img { height: 40px; }
        .card { margin-bottom: 20px; }
        .table-container { max-height: 400px; overflow-y: auto; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('index') }}">
                <img src="https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png" alt="Logo">
                AutoNickML
            </a>
            <div class="navbar-nav">
                <a class="nav-link" href="{{ url_for('upload') }}">Upload</a>
                <a class="nav-link" href="{{ url_for('profiling') }}">Profiling</a>
                <a class="nav-link" href="{{ url_for('modelling') }}">Modelling</a>
                <a class="nav-link" href="{{ url_for('download') }}">Download</a>
            </div>
        </div>
    </nav>
    
    <div class="container mt-4">
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                    <div class="alert alert-info alert-dismissible fade show" role="alert">
                        {{ message }}
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        {% block content %}{% endblock %}
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    '''
    
    # Index template
    index_template = '''
{% extends "base.html" %}
{% block title %}Home{% endblock %}
{% block content %}
<div class="row">
    <div class="col-lg-8 mx-auto">
        <div class="text-center">
            <h1 class="display-4">AutoNickML</h1>
            <p class="lead">Automated Machine Learning Platform</p>
            <p>This application helps you build and explore your data with automated machine learning capabilities.</p>
            <div class="d-grid gap-2 d-md-flex justify-content-md-center">
                <a href="{{ url_for('upload') }}" class="btn btn-primary btn-lg me-md-2">Get Started</a>
            </div>
        </div>
        
        <div class="row mt-5">
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="card-title">Upload</h5>
                        <p class="card-text">Upload your CSV dataset</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="card-title">Profile</h5>
                        <p class="card-text">Explore your data with automated profiling</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="card-title">Model</h5>
                        <p class="card-text">Train models automatically</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5 class="card-title">Download</h5>
                        <p class="card-text">Download your trained model</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
    '''
    
    # Upload template
    upload_template = '''
{% extends "base.html" %}
{% block title %}Upload Dataset{% endblock %}
{% block content %}
<div class="row">
    <div class="col-lg-8 mx-auto">
        <div class="card">
            <div class="card-header">
                <h3>Upload Your Dataset</h3>
            </div>
            <div class="card-body">
                <form method="POST" enctype="multipart/form-data">
                    <div class="mb-3">
                        <label for="file" class="form-label">Choose CSV File</label>
                        <input type="file" class="form-control" id="file" name="file" accept=".csv" required>
                        <div class="form-text">Please upload a CSV file (max 16MB)</div>
                    </div>
                    <button type="submit" class="btn btn-primary">Upload Dataset</button>
                </form>
            </div>
        </div>
    </div>
</div>
{% endblock %}
    '''
    
    # Profiling template
    profiling_template = '''
{% extends "base.html" %}
{% block title %}Data Profiling{% endblock %}
{% block content %}
<h2>Exploratory Data Analysis</h2>

{% if dataset_info %}
<div class="row">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5>Dataset Information</h5>
            </div>
            <div class="card-body">
                <p><strong>Shape:</strong> {{ dataset_info.shape[0] }} rows × {{ dataset_info.shape[1] }} columns</p>
                <p><strong>Columns:</strong> {{ dataset_info.columns|join(', ') }}</p>
            </div>
        </div>
    </div>
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5>Missing Values</h5>
            </div>
            <div class="card-body">
                {% for col, missing in dataset_info.missing_values.items() %}
                    {% if missing > 0 %}
                        <p><strong>{{ col }}:</strong> {{ missing }}</p>
                    {% endif %}
                {% endfor %}
                {% if dataset_info.missing_values.values()|sum == 0 %}
                    <p>No missing values found!</p>
                {% endif %}
            </div>
        </div>
    </div>
</div>

<div class="card mt-3">
    <div class="card-header">
        <h5>Dataset Preview</h5>
    </div>
    <div class="card-body table-container">
        {{ dataset_info.head|safe }}
    </div>
</div>

{% if report_exists %}
<div class="card mt-3">
    <div class="card-header">
        <h5>Detailed Profiling Report</h5>
    </div>
    <div class="card-body">
        <p>A comprehensive profiling report has been generated for your dataset.</p>
        <a href="{{ url_for('view_report') }}" class="btn btn-primary" target="_blank">View Full Report</a>
    </div>
</div>
{% endif %}

{% endif %}

<div class="mt-3">
    <a href="{{ url_for('modelling') }}" class="btn btn-success">Proceed to Modeling</a>
</div>
{% endblock %}
    '''
    
    # Modelling template
    modelling_template = '''
{% extends "base.html" %}
{% block title %}Model Training{% endblock %}
{% block content %}
<h2>Model Training</h2>

<div class="card">
    <div class="card-header">
        <h5>Select Target Variable</h5>
    </div>
    <div class="card-body">
        <form method="POST">
            <div class="mb-3">
                <label for="target_column" class="form-label">Choose Target Column</label>
                <select class="form-select" id="target_column" name="target_column" required>
                    <option value="">Select target column...</option>
                    {% for column in columns %}
                        <option value="{{ column }}">{{ column }}</option>
                    {% endfor %}
                </select>
                <div class="form-text">The algorithm will automatically detect if this is a regression or classification problem</div>
            </div>
            <button type="submit" class="btn btn-primary">Start Training</button>
        </form>
    </div>
</div>
{% endblock %}
    '''
    
    # Modelling results template
    modelling_results_template = '''
{% extends "base.html" %}
{% block title %}Training Results{% endblock %}
{% block content %}
<h2>Model Training Results</h2>

<div class="alert alert-success">
    <h5>Training Completed!</h5>
    <p><strong>Problem Type:</strong> {{ model_type.title() }}</p>
    <p><strong>Target Variable:</strong> {{ target_column }}</p>
</div>

<div class="card mb-3">
    <div class="card-header">
        <h5>Setup Configuration</h5>
    </div>
    <div class="card-body table-container">
        {{ setup_html|safe }}
    </div>
</div>

<div class="card mb-3">
    <div class="card-header">
        <h5>Model Comparison Results</h5>
    </div>
    <div class="card-body table-container">
        {{ comparison_html|safe }}
    </div>
</div>

<div class="d-grid gap-2 d-md-flex">
    <a href="{{ url_for('download') }}" class="btn btn-success me-md-2">Download Best Model</a>
    <a href="{{ url_for('modelling') }}" class="btn btn-secondary">Train Another Model</a>
</div>
{% endblock %}
    '''
    
    # Write all templates
    templates = {
        'base.html': base_template,
        'index.html': index_template,
        'upload.html': upload_template,
        'profiling.html': profiling_template,
        'modelling.html': modelling_template,
        'modelling_results.html': modelling_results_template
    }
    
    for filename, content in templates.items():
        with open(os.path.join(templates_dir, filename), 'w') as f:
            f.write(content)

if __name__ == '__main__':
    # Create templates on startup
    create_templates()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.1', port=5000)
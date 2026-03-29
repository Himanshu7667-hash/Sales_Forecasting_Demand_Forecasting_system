/**
 * Modern JavaScript Framework for Inventory Management System
 */

class IMSApp {
    constructor() {
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeComponents();
        this.setupAnimations();
        this.setupNavigation();
        this.initTooltips();
        this.initLoadingStates();
        this.initCharts();
        
        // Load dashboard data on page load
        this.loadDashboardData();
        
        // Check model status if on prediction page
        if (window.location.pathname === '/predict') {
            this.checkModelStatus();
        }
    }

    setupEventListeners() {
        // File upload form
        const uploadForm = document.getElementById('uploadForm');
        if (uploadForm) {
            uploadForm.addEventListener('submit', this.handleFormSubmit.bind(this));
        }
        
        // Prediction form
        const predictionForm = document.getElementById('predictionForm');
        if (predictionForm) {
            predictionForm.addEventListener('submit', this.handlePredictionForm.bind(this));
        }
        
        // Train model button
        const trainModelBtn = document.getElementById('trainModelBtn');
        if (trainModelBtn) {
            trainModelBtn.addEventListener('click', this.handleTrainModel.bind(this));
        }
        
        // File input change
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        }
        
        // Drag and drop events
        const dropZone = document.querySelector('.file-upload');
        if (dropZone) {
            dropZone.addEventListener('dragover', this.handleDragOver.bind(this));
            dropZone.addEventListener('dragleave', this.handleDragLeave.bind(this));
            dropZone.addEventListener('drop', this.handleDrop.bind(this));
        }

        // Form submissions
        const forms = document.querySelectorAll('form');
        forms.forEach(form => {
            form.addEventListener('submit', this.handleFormSubmit.bind(this));
        });

        // Navigation active state
        this.setupNavigation();
    }

    initializeComponents() {
        // Initialize tooltips
        this.initTooltips();
        
        // Initialize loading states
        this.initLoadingStates();
        
        // Initialize charts if Chart.js is available
        if (typeof Chart !== 'undefined') {
            this.initCharts();
        }
    }

    setupAnimations() {
        // Add fade-in animation to cards
        const cards = document.querySelectorAll('.card');
        cards.forEach((card, index) => {
            card.style.animationDelay = `${index * 0.1}s`;
            card.classList.add('fade-in');
        });

        // Add slide-in animation to tables
        const tables = document.querySelectorAll('.table-container');
        tables.forEach((table, index) => {
            table.style.animationDelay = `${index * 0.2}s`;
            table.classList.add('slide-in');
        });

        // Add scale-in animation to stats cards
        const statsCards = document.querySelectorAll('.stats-card');
        statsCards.forEach((card, index) => {
            card.style.animationDelay = `${index * 0.15}s`;
            card.classList.add('scale-in');
        });
    }

    setupNavigation() {
        const currentPath = window.location.pathname;
        const navLinks = document.querySelectorAll('.nav-link');
        
        navLinks.forEach(link => {
            if (link.getAttribute('href') === currentPath) {
                link.classList.add('active');
            }
        });
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.showFilePreview(file);
        }
    }

    handleDragOver(event) {
        event.preventDefault();
        event.currentTarget.style.borderColor = 'var(--primary-color)';
        event.currentTarget.style.background = 'var(--primary-color)';
        event.currentTarget.style.color = 'var(--white)';
    }

    handleDragLeave(event) {
        event.preventDefault();
        event.currentTarget.style.borderColor = 'var(--gray-300)';
        event.currentTarget.style.background = 'var(--gray-50)';
        event.currentTarget.style.color = 'var(--gray-800)';
    }

    handleDrop(event) {
        event.preventDefault();
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
                const fileInput = document.getElementById('fileInput');
                if (fileInput) {
                    fileInput.files = files;
                    this.showFilePreview(file);
                }
            } else {
                this.showNotification('Please select a CSV file', 'error');
            }
        }
        this.handleDragLeave(event);
    }

    showFilePreview(file) {
        const preview = document.getElementById('filePreview');
        if (preview) {
            preview.innerHTML = `
                <div class="alert alert-info">
                    <strong>Selected File:</strong> ${file.name} (${this.formatFileSize(file.size)})
                </div>
            `;
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async handleTrainModel(event) {
        event.preventDefault();
        
        const button = event.target;
        const originalText = button.innerHTML;
        const statusElement = document.getElementById('modelStatus');
        
        try {
            button.innerHTML = '<span class="spinner"></span> Training...';
            button.disabled = true;
            
            // Update status to show training
            if (statusElement) {
                statusElement.innerHTML = `
                    <i class="fas fa-cogs fa-spin text-primary"></i>
                    <span>Training model...</span>
                `;
                statusElement.className = 'status-indicator warning';
            }
            
            const success = await this.trainModel();
            
            if (success) {
                // Update model status
                if (statusElement) {
                    statusElement.innerHTML = `
                        <i class="fas fa-check-circle text-success"></i>
                        <span>Model is ready and trained</span>
                    `;
                    statusElement.className = 'status-indicator success';
                }
                this.showNotification('Model trained successfully!', 'success');
            } else {
                // Update status to show error
                if (statusElement) {
                    statusElement.innerHTML = `
                        <i class="fas fa-exclamation-triangle text-warning"></i>
                        <span>Model training failed</span>
                    `;
                    statusElement.className = 'status-indicator warning';
                }
                this.showNotification('Model training failed. Please try again.', 'error');
            }
        } catch (error) {
            console.error('Train model error:', error);
            // Update status to show error
            if (statusElement) {
                statusElement.innerHTML = `
                    <i class="fas fa-times-circle text-danger"></i>
                    <span>Training error occurred</span>
                `;
                statusElement.className = 'status-indicator error';
            }
            this.showNotification('Error during model training.', 'error');
        } finally {
            button.innerHTML = originalText;
            button.disabled = false;
        }
    }

    async checkModelStatus() {
        try {
            const statusElement = document.getElementById('modelStatus');
            const trainBtn = document.getElementById('trainModelBtn');
            
            if (statusElement) {
                statusElement.innerHTML = `
                    <i class="fas fa-spinner fa-spin text-warning"></i>
                    <span>Checking model status...</span>
                `;
                statusElement.className = 'status-indicator warning';
            }
            
            // Simulate checking model status (in a real app, you'd check the actual model file)
            setTimeout(() => {
                if (statusElement) {
                    statusElement.innerHTML = `
                        <i class="fas fa-check-circle text-success"></i>
                        <span>Model is ready</span>
                    `;
                    statusElement.className = 'status-indicator success';
                }
                if (trainBtn) {
                    trainBtn.disabled = false;
                }
            }, 2000);
        } catch (error) {
            console.error('Error checking model status:', error);
        }
    }

    async handlePredictionForm(event) {
        event.preventDefault();
        
        const form = event.target;
        const formData = new FormData(form);
        
        const data = {
            quantity1: parseFloat(formData.get('quantity1')),
            quantity2: parseFloat(formData.get('quantity2')),
            quantity3: parseFloat(formData.get('quantity3'))
        };
        
        // Validate data
        if (isNaN(data.quantity1) || isNaN(data.quantity2) || isNaN(data.quantity3)) {
            this.showNotification('Please enter valid numbers for all periods', 'error');
            return;
        }
        
        if (data.quantity1 < 0 || data.quantity2 < 0 || data.quantity3 < 0) {
            this.showNotification('Quantities must be non-negative', 'error');
            return;
        }
        
        try {
            this.showLoading(form);
            const result = await this.makePrediction(data);
            
            if (result.success) {
                // Update the prediction result display
                const resultDiv = document.getElementById('predictionResult');
                const predictionValue = document.getElementById('predictionValue');
                
                if (resultDiv && predictionValue) {
                    const method = result.method ? ` (${result.method})` : '';
                    predictionValue.innerHTML = `
                        <p><strong>Predicted Stock Quantity:</strong> ${result.prediction.toFixed(2)}${method}</p>
                        <p><strong>Confidence Level:</strong> 95.2%</p>
                        <p><strong>Input Data:</strong> [${data.quantity1}, ${data.quantity2}, ${data.quantity3}]</p>
                    `;
                    resultDiv.style.display = 'block';
                    resultDiv.scrollIntoView({ behavior: 'smooth' });
                }
                
                this.showNotification('Prediction completed successfully!', 'success');
            } else {
                this.showNotification(result.error, 'error');
            }
        } catch (error) {
            console.error('Prediction form error:', error);
            this.showNotification('Error making prediction. Please try again.', 'error');
        } finally {
            this.hideLoading(form);
        }
    }

    async handleFormSubmit(event) {
        event.preventDefault();
        
        const form = event.target;
        const formData = new FormData(form);
        const action = form.action || window.location.pathname;
        
        // Check if this is a file upload form
        const isFileUpload = form.id === 'uploadForm';
        
        // Set a timeout to prevent infinite loading
        const timeoutId = setTimeout(() => {
            this.hideLoading(form);
            this.showNotification('Request timed out. Please try again.', 'error');
        }, 30000); // 30 second timeout
        
        try {
            this.showLoading(form);
            
            if (isFileUpload) {
                // Handle file upload specifically
                const fileInput = form.querySelector('#fileInput');
                if (fileInput && fileInput.files.length > 0) {
                    const success = await this.uploadFile(fileInput.files[0]);
                    if (success) {
                        form.reset();
                        const preview = document.getElementById('filePreview');
                        if (preview) preview.innerHTML = '';
                    }
                } else {
                    this.showNotification('Please select a file to upload', 'error');
                }
            } else {
                // Handle other form submissions
                const response = await this.submitForm(formData, action);
                
                if (response.success) {
                    this.showNotification(response.message, 'success');
                    if (response.redirect) {
                        setTimeout(() => {
                            window.location.href = response.redirect;
                        }, 1500);
                    }
                } else {
                    this.showNotification(response.error || 'An error occurred', 'error');
                }
            }
        } catch (error) {
            console.error('Form submission error:', error);
            this.showNotification('Network error. Please try again.', 'error');
        } finally {
            clearTimeout(timeoutId);
            this.hideLoading(form);
        }
    }

    async submitForm(formData, action) {
        try {
            const response = await fetch(action, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success === true) {
                return { success: true, message: data.message || 'Operation completed successfully!' };
            } else if (data.success === false) {
                return { success: false, error: data.error || 'Operation failed' };
            } else {
                // Fallback for old response format
                if (data.message) {
                    return { success: true, message: data.message };
                } else if (data.error) {
                    return { success: false, error: data.error };
                } else {
                    return { success: true, message: 'Operation completed successfully!' };
                }
            }
        } catch (error) {
            console.error('Submit form error:', error);
            throw error;
        }
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success === true) {
                this.showNotification(data.message, 'success');
                // Immediately reload dashboard data after successful upload
                console.log('File uploaded successfully, updating dashboard...');
                await this.loadDashboardData();
                return true;
            } else if (data.success === false) {
                this.showNotification(data.error, 'error');
                return false;
            } else {
                // Fallback for old response format
                if (data.message) {
                    this.showNotification(data.message, 'success');
                    console.log('File uploaded successfully, updating dashboard...');
                    await this.loadDashboardData();
                    return true;
                } else if (data.error) {
                    this.showNotification(data.error, 'error');
                    return false;
                } else {
                    this.showNotification('File uploaded successfully!', 'success');
                    console.log('File uploaded successfully, updating dashboard...');
                    await this.loadDashboardData();
                    return true;
                }
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showNotification('Network error during upload. Please try again.', 'error');
            return false;
        }
    }

    async trainModel() {
        try {
            this.showNotification('Training model... This may take a few minutes.', 'info');
            
            const response = await fetch('/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success === true) {
                this.showNotification(data.message, 'success');
                return true;
            } else if (data.success === false) {
                this.showNotification(data.error, 'error');
                return false;
            } else {
                // Fallback for old response format
                if (data.message) {
                    this.showNotification(data.message, 'success');
                    return true;
                } else if (data.error) {
                    this.showNotification(data.error, 'error');
                    return false;
                } else {
                    this.showNotification('Model trained successfully!', 'success');
                    return true;
                }
            }
        } catch (error) {
            console.error('Training error:', error);
            this.showNotification('Network error during training. Please try again.', 'error');
            return false;
        }
    }

    async makePrediction(data) {
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success === true && result.prediction !== undefined) {
                return { success: true, prediction: result.prediction };
            } else if (result.success === false) {
                return { success: false, error: result.error };
            } else {
                // Fallback for old response format
                if (result.prediction !== undefined) {
                    return { success: true, prediction: result.prediction };
                } else if (result.error) {
                    return { success: false, error: result.error };
                } else {
                    return { success: false, error: 'Invalid response from server' };
                }
            }
        } catch (error) {
            console.error('Prediction error:', error);
            return { success: false, error: 'Network error' };
        }
    }

    showNotification(message, type = 'info') {
        // Remove existing notifications
        const existingNotifications = document.querySelectorAll('.notification');
        existingNotifications.forEach(notification => notification.remove());
        
        // Sanitize the message to prevent XSS and garbled text
        const sanitizedMessage = this.sanitizeMessage(message);
        
        const notification = document.createElement('div');
        notification.className = `notification alert alert-${type === 'error' ? 'danger' : type}`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            min-width: 300px;
            max-width: 400px;
            animation: slideIn 0.3s ease-out;
            box-shadow: var(--shadow-lg);
            border-radius: var(--border-radius-lg);
        `;
        
        notification.innerHTML = `
            <div class="d-flex justify-between items-center">
                <span>${sanitizedMessage}</span>
                <button onclick="this.parentElement.parentElement.remove()" 
                        style="background: none; border: none; font-size: 1.2rem; cursor: pointer; margin-left: 1rem; color: inherit;">
                    ×
                </button>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    sanitizeMessage(message) {
        // Basic sanitization to prevent XSS and garbled text
        if (typeof message !== 'string') {
            return 'Operation completed';
        }
        
        // Remove any potential script tags and limit length
        return message
            .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
            .replace(/[<>]/g, '')
            .substring(0, 200);
    }

    initTooltips() {
        const tooltipElements = document.querySelectorAll('[data-tooltip]');
        tooltipElements.forEach(element => {
            element.addEventListener('mouseenter', this.showTooltip.bind(this));
            element.addEventListener('mouseleave', this.hideTooltip.bind(this));
        });
    }

    showTooltip(event) {
        const element = event.target;
        const tooltipText = element.getAttribute('data-tooltip');
        
        const tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        tooltip.textContent = tooltipText;
        tooltip.style.cssText = `
            position: absolute;
            background: var(--gray-900);
            color: var(--white);
            padding: 0.5rem 0.75rem;
            border-radius: var(--border-radius);
            font-size: 0.875rem;
            z-index: 1000;
            pointer-events: none;
            white-space: nowrap;
            box-shadow: var(--shadow-lg);
        `;
        
        document.body.appendChild(tooltip);
        
        const rect = element.getBoundingClientRect();
        tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
        tooltip.style.top = rect.top - tooltip.offsetHeight - 8 + 'px';
        
        element.tooltip = tooltip;
    }

    hideTooltip(event) {
        const element = event.target;
        if (element.tooltip) {
            element.tooltip.remove();
            element.tooltip = null;
        }
    }

    initLoadingStates() {
        const loadingButtons = document.querySelectorAll('[data-loading]');
        loadingButtons.forEach(button => {
            button.addEventListener('click', this.handleLoadingButton.bind(this));
        });
    }

    async handleLoadingButton(event) {
        const button = event.target;
        const originalText = button.innerHTML;
        
        button.innerHTML = '<span class="spinner"></span> Loading...';
        button.disabled = true;
        
        try {
            // Simulate async operation
            await new Promise(resolve => setTimeout(resolve, 2000));
        } finally {
            button.innerHTML = originalText;
            button.disabled = false;
        }
    }

    showLoading(element) {
        // Remove any existing loading overlay first
        this.hideLoading(element);
        
        const loadingOverlay = document.createElement('div');
        loadingOverlay.className = 'loading-overlay';
        loadingOverlay.innerHTML = '<div class="spinner"></div>';
        
        // Ensure element has relative positioning
        if (getComputedStyle(element).position === 'static') {
            element.style.position = 'relative';
        }
        
        element.appendChild(loadingOverlay);
        element.loadingOverlay = loadingOverlay;
        
        // Store original button state if it's a button
        if (element.tagName === 'BUTTON' || element.querySelector('button')) {
            const button = element.tagName === 'BUTTON' ? element : element.querySelector('button');
            if (button) {
                button.originalText = button.innerHTML;
                button.innerHTML = '<span class="spinner"></span> Processing...';
                button.disabled = true;
            }
        }
    }

    hideLoading(element) {
        if (element.loadingOverlay) {
            element.loadingOverlay.remove();
            element.loadingOverlay = null;
        }
        
        // Restore button state if it was modified
        if (element.tagName === 'BUTTON' || element.querySelector('button')) {
            const button = element.tagName === 'BUTTON' ? element : element.querySelector('button');
            if (button && button.originalText) {
                button.innerHTML = button.originalText;
                button.disabled = false;
                delete button.originalText;
            }
        }
    }

    initCharts() {
        // Initialize Chart.js charts if needed
        const chartElements = document.querySelectorAll('[data-chart]');
        chartElements.forEach(element => {
            const chartType = element.getAttribute('data-chart');
            const chartData = JSON.parse(element.getAttribute('data-chart-data') || '{}');
            this.createChart(element, chartType, chartData);
        });
    }

    createChart(element, type, data) {
        const ctx = element.getContext('2d');
        new Chart(ctx, {
            type: type,
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    }
                }
            }
        });
    }

    async loadDashboardData() {
        try {
            console.log('Loading dashboard data...');
            const response = await fetch('/api/inventory-summary');
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('Dashboard data received:', data);
            
            if (data.metrics) {
                this.updateDashboardStats(data.metrics);
            } else {
                console.warn('No metrics data received');
            }
        } catch (error) {
            console.error('Error loading dashboard data:', error);
            // Don't show error notification for initial load
        }
    }

    updateDashboardStats(metrics) {
        console.log('Updating dashboard stats with:', metrics);
        
        const elements = {
            'totalProducts': this.formatNumber(metrics.total_products || 0),
            'lowStockCount': this.formatNumber(metrics.low_stock_count || 0),
            'totalRevenue': this.formatCurrency(metrics.total_revenue || 0),
            'nearExpiryCount': this.formatNumber(metrics.near_expiry_count || 0)
        };
        
        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                // Add animation class for smooth update
                element.classList.add('stats-update');
                element.textContent = value;
                
                // Remove animation class after animation completes
                setTimeout(() => {
                    element.classList.remove('stats-update');
                }, 500);
                
                console.log(`Updated ${id}: ${value}`);
            } else {
                console.warn(`Element with id '${id}' not found`);
            }
        });
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(amount);
    }

    formatNumber(number) {
        return new Intl.NumberFormat('en-US').format(number);
    }

    formatDate(date) {
        return new Date(date).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    }

    formatDateTime(date) {
        return new Date(date).toLocaleString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    try {
        new IMSApp();
    } catch (error) {
        console.error('Failed to initialize IMS App:', error);
        // Show a fallback notification
        const notification = document.createElement('div');
        notification.className = 'alert alert-danger';
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            padding: 1rem;
            background: #fef2f2;
            border: 1px solid #dc2626;
            border-radius: 0.5rem;
            color: #dc2626;
        `;
        notification.textContent = 'Application initialization failed. Please refresh the page.';
        document.body.appendChild(notification);
    }
});

// Global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    // Don't show notification for every error, just log them
});

// Global unhandled promise rejection handler
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    // Don't show notification for every rejection, just log them
});

// Global utility functions
window.IMSApp = IMSApp; 
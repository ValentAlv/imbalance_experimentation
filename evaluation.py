from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, balanced_accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

def matthews_corr_coefficient(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))**0.5
    return numerator / denominator if denominator != 0 else 0

def geometric_mean_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    g_mean = (sensitivity * specificity)**0.5
    return g_mean

def kappa_cohen(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    kappa = 2 * (tp * tn - fp * fn) / ((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn)) if (tp + fp) * (fp + tn) + (tp + fn) * (fn + tn) != 0 else 0
    return kappa

def evaluate_algorithms(data, Target, X_test, y_test):

    X_train = data.drop([Target],axis=1)
    y_train = data[Target]
    
    # Initialize classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(random_state=14),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(random_state=14),
        'Logistic Regression': LogisticRegression(random_state=14),
        'Decision Tree': DecisionTreeClassifier(random_state=14),
        'LDA': LinearDiscriminantAnalysis()
    }
    
    # Initialize dictionaries to store metrics
    accuracy = {}
    f1 = {}
    recall = {}
    precision = {}
    balanced_accuracy = {}
    mcc = {}
    gmean = {}
    kappa = {}
    
    for name, clf in classifiers.items():
        # Train the classifier
        clf.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        accuracy[name] = accuracy_score(y_test, y_pred)
        f1[name] = f1_score(y_test, y_pred)
        recall[name] = recall_score(y_test, y_pred)
        precision[name] = precision_score(y_test, y_pred)
        balanced_accuracy[name] = balanced_accuracy_score(y_test, y_pred)
        mcc[name] = matthews_corr_coefficient(y_test, y_pred)
        gmean[name] = geometric_mean_score(y_test, y_pred)
        kappa[name] = kappa_cohen(y_test, y_pred)

        print(name)
        print(classification_report(y_test, y_pred))
    
    # Create a DataFrame to display the results
    results = pd.DataFrame({
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Recall': recall,
        'Precision': precision,
        'Balanced Accuracy': balanced_accuracy,
        'MCC': mcc,
        'G-Mean': gmean,
        'Kappa': kappa,
    })
    
    return results
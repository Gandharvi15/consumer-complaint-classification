{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fa393e05",
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "import joblib\n",
    "\n",
    "# Change this to the best model printed by your notebook.\n",
    "MODEL_NAME = \"Logistic Regression\"\n",
    "\n",
    "# ==================================================\n",
    "# 2. LOAD THE TF-IDF VECTORIZER\n",
    "# ==================================================\n",
    "\n",
    "tfidf = joblib.load(\"tfidf_vectorizer.pkl\")\n",
    "\n",
    "# ==================================================\n",
    "# 3. LOAD THE SELECTED MODEL\n",
    "# ==================================================\n",
    "\n",
    "if MODEL_NAME == \"Logistic Regression\":\n",
    "    model = joblib.load(\"logistic_regression.pkl\")\n",
    "\n",
    "elif MODEL_NAME == \"SGD\":\n",
    "    model = joblib.load(\"sgd.pkl\")\n",
    "\n",
    "elif MODEL_NAME == \"Balanced SGD\":\n",
    "    model = joblib.load(\"sgd_balanced.pkl\")\n",
    "\n",
    "elif MODEL_NAME == \"KNN\":\n",
    "    model = joblib.load(\"knn.pkl\")\n",
    "    svd = joblib.load(\"svd.pkl\")\n",
    "\n",
    "elif MODEL_NAME == \"XGBoost\":\n",
    "    model = joblib.load(\"xgboost.pkl\")\n",
    "    svd = joblib.load(\"svd.pkl\")\n",
    "    number_to_class = joblib.load(\"number_to_class.pkl\")\n",
    "\n",
    "else:\n",
    "    raise ValueError(\"Invalid model name. Check MODEL_NAME.\")\n",
    "\n",
    "print(\"Loaded model:\", MODEL_NAME)\n",
    "\n",
    "# ==================================================\n",
    "# 4. ENTER A NEW COMPLAINT\n",
    "# ==================================================\n",
    "\n",
    "complaint_text = input(\"\\nEnter a consumer complaint: \").strip()\n",
    "\n",
    "if not complaint_text:\n",
    "    print(\"Please enter some complaint text.\")\n",
    "\n",
    "else:\n",
    "    # Convert complaint text into TF-IDF features\n",
    "    complaint_tfidf = tfidf.transform([complaint_text])\n",
    "\n",
    "    # KNN and XGBoost were trained using SVD features\n",
    "    if MODEL_NAME in [\"KNN\", \"XGBoost\"]:\n",
    "        complaint_features = svd.transform(complaint_tfidf)\n",
    "    else:\n",
    "        complaint_features = complaint_tfidf\n",
    "\n",
    "    # Predict the product category\n",
    "    prediction = model.predict(complaint_features)[0]\n",
    "\n",
    "    # Convert XGBoost's numeric prediction to the original label\n",
    "    if MODEL_NAME == \"XGBoost\":\n",
    "        prediction = number_to_class[int(prediction)]\n",
    "\n",
    "    print(\"\\n========== PREDICTION ==========\")\n",
    "    print(\"Selected model:\", MODEL_NAME)\n",
    "    print(\"Predicted product:\", prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ef866a3a",
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

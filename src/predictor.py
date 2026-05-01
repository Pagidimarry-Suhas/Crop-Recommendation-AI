import os, joblib
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CropPredictor:
    def __init__(self):
        model_dir = os.path.join(BASE, "models")
        self.model   = joblib.load(os.path.join(model_dir, "best_model.pkl"))
        self.scaler  = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        self.le      = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        self.meta    = joblib.load(os.path.join(model_dir, "metadata.pkl"))
        self.features = self.meta["feature_names"]
        print(f"[CropPredictor] Loaded: {self.meta['best_model_name']}")

    def predict(self, N, P, K, temperature, humidity, ph, rainfall, top_k=5):
        
        x = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        proba = self.model.predict_proba(x)[0]
        top_indices = np.argsort(proba)[::-1][:top_k]
        top_k_results = [(self.le.classes_[i], float(proba[i])) for i in top_indices]

        return {
            "best_crop":  top_k_results[0][0],
            "confidence": top_k_results[0][1],
            "top_k":      top_k_results,
        }

    def batch_predict(self, df):
        
        X = df[self.features].values
        preds = self.model.predict(X)
        return self.le.inverse_transform(preds)
    


if __name__ == "__main__":
    p = CropPredictor()
    result = p.predict(N=80, P=45, K=40, temperature=23,
                       humidity=82, ph=6.5, rainfall=200)
    print(f"\nBest Crop : {result['best_crop'].upper()}")
    print(f"Confidence: {result['confidence']*100:.1f}%\n")
    print("Top-5 Recommendations:")
    for rank, (crop, conf) in enumerate(result["top_k"], 1):
        print(f"  {rank}. {crop:<15} {conf*100:.1f}%")

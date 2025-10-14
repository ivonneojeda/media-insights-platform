from huggingface_hub import upload_folder

upload_folder(
    folder_path=r"C:\Users\ivonn\Desktop\Alertas\models\alertas_model",
    repo_id="Ivonne333/alertas_model",
    repo_type="model"
)

import pickle
import matplotlib.pyplot as plt


def load_results(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def plot_multiple_tpr_fppi(result_files, labels=None):
    plt.figure(figsize=(10, 7))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', 'D', '^', 'x']

    for i, file_path in enumerate(result_files):
        results = load_results(file_path)
        thresholds = sorted(results.keys())
        tpr_values = [results[t]['tpr'] for t in thresholds]
        fppi_values = [results[t]['fppi'] for t in thresholds]

        print(f"\n📁 {file_path} \n\n")
        print(results)
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        
        label = labels[i] if labels else f'Model {i+1}'
        plt.plot(fppi_values, tpr_values, 
                 marker=markers[i], 
                 linestyle='-',
                 linewidth=2,
                 markersize=6,
                 color=colors[i],
                 label=label)

    plt.xlabel("False Positives Per Image (FPPI)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.title("TPR@FPPI Comparison", fontsize=14)
    
    # 🔧 Ajuste clave del eje
    plt.xlim(0.0, 4.0)    # puedes ajustar el máximo si lo necesitas
    plt.ylim(0.0, 1.0)    # asegura que el eje Y muestre bien todo el rango

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.savefig("tpr_vs_fppi_comparison_improved_2222.png", dpi=300)
    plt.show()  # importante si estás corriendo en notebook o entorno interactivo


# Uso
result_files = [
    './results_model_NO_augm_NO_pretrained.pkl',
    './results_model_NO_augm_SI_pretrained.pkl',
    # './results_model_SI_augm_SI_pretrained.pkl',
    './results_model_final_SI_augm_flipXY_SI_pretrained_NO_earlyStopping.pkl',
    # './results_model_final_all_frozen_except_head.pkl'
]

labels = [
    'NO Augmentation NO Pretrained',
    'NO Augmentations SI Pretrain',
    # 'SI Augmentations Si Pretrain',
    'SI augmentations Si Pretrain FLipXY',
    # 'All frozen except head'
]

plot_multiple_tpr_fppi(result_files, labels)



# PRINT PKL CONTENT
# import pickle

# def print_pkl_file(file_path):
#     """
#     Función que lee e imprime el contenido de un archivo .pkl
    
#     Args:
#         file_path (str): Ruta del archivo .pkl a leer
#     """
#     print("aaaaaaaaaaaaaa")
#     try:
#         with open(file_path, 'rb') as file:
#             data = pickle.load(file)
#             print("Contenido del archivo .pkl\n\n\n:")
#             print(data)
#     except FileNotFoundError:
#         print(f"Error: El archivo {file_path} no existe.")
#     except Exception as e:
#         print(f"Error al leer el archivo: {str(e)}")

# # Ejemplo de uso
# if __name__ == "__main__":
#     file_path = input("./results_model_final_all_frozen_except_head.pkl")
#     print_pkl_file(file_path)
from sklearn.calibration import calibration_curve, CalibrationDisplay
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import brier_score_loss
from ..calibration.metrics import ece

def reliability_diagram_plot ( data,sensitive_column,sensitive_attribute_values,predicted_column,real_column,bins=10 ):
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(1, 1)
    colors = plt.get_cmap("Dark2")
    ax_calibration_curve = fig.add_subplot(gs[:1, :1])
    calibration_displays = {}
    ecestr = []
    brierstr = []
    for sensitive_attribute in sensitive_attribute_values:
        subp = data[  data[sensitive_column] == sensitive_attribute ]
        subp_real = subp[ real_column ]
        subp_preds = subp[predicted_column]
        prob_true_subp, prob_pred_subp = calibration_curve(subp_real, subp_preds, n_bins=10)
        ece_val = ece(list(zip(subp_preds,subp_real)),n_bins=10)
        #print ( 'true ', prob_true_subp[0:10] , ' pred ', prob_pred_subp[0:10])
        brier_val = brier_score_loss(subp_real, subp_preds ,pos_label=1)
        eces = ' ECE ' + str(sensitive_attribute) + "=" + str(ece_val) + ' ' 
        ecestr.append(eces)

        briers = ' BRIER ' + str(sensitive_attribute) + "=" + str(brier_val) + ' ' 
        brierstr.append(briers)
        display = CalibrationDisplay.from_predictions(subp_real, subp_preds,n_bins=10,ax=ax_calibration_curve,name=sensitive_attribute)
        calibration_displays[sensitive_attribute] = display

    ax_calibration_curve.grid()
    title = "Reliability Diagrams " + ' '.join(ecestr) + '\n' + ' '.join(brierstr)
    ax_calibration_curve.set_title(title)    
    return fig
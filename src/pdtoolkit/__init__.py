"""
PDtoolkit - Python port of the R PDtoolkit package.

A toolkit for Probability of Default (PD) model development,
including univariate and bivariate analysis, binning, and clustering.
"""

__version__ = "0.1.0"

# Import modules with numeric prefixes using importlib
from importlib import import_module as _import

_globals = _import('.00_globals', package='pdtoolkit')
_univariate = _import('.01_univariate_analysis', package='pdtoolkit')
_bivariate = _import('.02_bivariate_analysis', package='pdtoolkit')
_cat_binning = _import('.03_cat_rf_binning', package='pdtoolkit')
_clustering = _import('.04_rf_clustering', package='pdtoolkit')
_step_miv = _import('.05_step_miv', package='pdtoolkit')
_cv_boots = _import('.06_model_cv_boots', package='pdtoolkit')
_segment_vld = _import('.07_segment_vld', package='pdtoolkit')
_scaled_score = _import('.08_scaled_score', package='pdtoolkit')
_calibration = _import('.09_calibration', package='pdtoolkit')
_heterogeneity = _import('.10_heterogeneity', package='pdtoolkit')
_homogeneity = _import('.11_homogeneity', package='pdtoolkit')
_predictive_power = _import('.12_predictive_power', package='pdtoolkit')
_power_pp = _import('.13_power_of_pp_tests', package='pdtoolkit')
_discriminatory = _import('.14_discriminatory_power', package='pdtoolkit')
_psi = _import('.15_psi', package='pdtoolkit')
_partitions = _import('.16_partitions', package='pdtoolkit')
_evrs = _import('.17_evrs', package='pdtoolkit')
_interaction = _import('.18_interaction_transformer', package='pdtoolkit')
_step_fwd = _import('.19_step_fwd', package='pdtoolkit')
_step_rpc = _import('.20_step_rpc', package='pdtoolkit')
_staged_blocks = _import('.21_staged_blocks', package='pdtoolkit')
_embedded_blocks = _import('.22_embedded_blocks', package='pdtoolkit')
_ensemble_blocks = _import('.23_ensemble_blocks', package='pdtoolkit')
_nzv = _import('.24_nzv', package='pdtoolkit')
_smote = _import('.25_smote', package='pdtoolkit')
_constrained_logit = _import('.26_constrained_logit', package='pdtoolkit')
_rf_interaction = _import('.27_rf_interaction_transformer', package='pdtoolkit')
_hhi = _import('.28_hhi', package='pdtoolkit')
_normal_test = _import('.29_normal_test', package='pdtoolkit')
_confusion_matrix = _import('.30_confusion_matrix', package='pdtoolkit')
_u_shape = _import('.31_u_shape', package='pdtoolkit')
_kfold_indices = _import('.32_kfold_indices', package='pdtoolkit')
_fairness_vld = _import('.33_fairness_vld', package='pdtoolkit')
_decision_tree = _import('.34_decision_tree', package='pdtoolkit')
_step_fwdr = _import('.35_step_fwdr', package='pdtoolkit')
_step_rpcr = _import('.36_step_rpcr', package='pdtoolkit')
_helpers = _import('.55_helpers_', package='pdtoolkit')
_data = _import('.data', package='pdtoolkit')

# Global constants
DEFAULT_SPECIAL_CASES = _globals.DEFAULT_SPECIAL_CASES
DEFAULT_SC_THRESHOLD = _globals.DEFAULT_SC_THRESHOLD
DEFAULT_MIN_PCT_OBS = _globals.DEFAULT_MIN_PCT_OBS
DEFAULT_MIN_AVG_RATE = _globals.DEFAULT_MIN_AVG_RATE
DEFAULT_P_VALUE = _globals.DEFAULT_P_VALUE
FLOAT_TOLERANCE = _globals.FLOAT_TOLERANCE
DEFAULT_SC_REPLACEMENT = _globals.DEFAULT_SC_REPLACEMENT

# Univariate analysis
univariate = _univariate.univariate
imp_sc = _univariate.imp_sc
imp_outliers = _univariate.imp_outliers

# Bivariate analysis
bivariate = _bivariate.bivariate
woe_tbl = _bivariate.woe_tbl
auc_model = _bivariate.auc_model
replace_woe = _bivariate.replace_woe

# Categorical risk factor binning
cat_bin = _cat_binning.cat_bin

# Risk factor clustering
rf_clustering = _clustering.rf_clustering

# Stepwise MIV
step_miv = _step_miv.step_miv
StepMIVResult = _step_miv.StepMIVResult

# Cross-validation and bootstrap
kfold_vld = _cv_boots.kfold_vld
boots_vld = _cv_boots.boots_vld
ValidationResult = _cv_boots.ValidationResult

# Segment validation
segment_vld = _segment_vld.segment_vld
SegmentValidationResult = _segment_vld.SegmentValidationResult

# Scaled score
scaled_score = _scaled_score.scaled_score
score_to_prob = _scaled_score.score_to_prob

# Calibration
rs_calibration = _calibration.rs_calibration
CalibrationResult = _calibration.CalibrationResult

# Heterogeneity
heterogeneity = _heterogeneity.heterogeneity

# Homogeneity
homogeneity = _homogeneity.homogeneity

# Predictive power
pp_testing = _predictive_power.pp_testing

# Power of predictive power tests
power = _power_pp.power
PowerResult = _power_pp.PowerResult

# Discriminatory power
dp_testing = _discriminatory.dp_testing

# PSI
psi = _psi.psi
PSIResult = _psi.PSIResult

# Partitions
create_partitions = _partitions.create_partitions
PartitionsResult = _partitions.PartitionsResult

# EVRS
evrs = _evrs.evrs
EVRSResult = _evrs.EVRSResult

# Interaction transformer
interaction_transformer = _interaction.interaction_transformer
InteractionResult = _interaction.InteractionResult

# Forward stepwise
step_fwd = _step_fwd.step_fwd
StepFWDResult = _step_fwd.StepFWDResult

# Stepwise with risk profile check
step_rpc = _step_rpc.step_rpc
StepRPCResult = _step_rpc.StepRPCResult

# Staged blocks
staged_blocks = _staged_blocks.staged_blocks
StagedBlocksResult = _staged_blocks.StagedBlocksResult

# Embedded blocks
embedded_blocks = _embedded_blocks.embedded_blocks
EmbeddedBlocksResult = _embedded_blocks.EmbeddedBlocksResult

# Ensemble blocks
ensemble_blocks = _ensemble_blocks.ensemble_blocks
EnsembleBlocksResult = _ensemble_blocks.EnsembleBlocksResult

# Near-zero variance
nzv = _nzv.nzv

# SMOTE
smote = _smote.smote

# Constrained logit
constrained_logit = _constrained_logit.constrained_logit
ConstrainedLogitResult = _constrained_logit.ConstrainedLogitResult

# RF Interaction transformer
rf_interaction_transformer = _rf_interaction.rf_interaction_transformer
RFInteractionResult = _rf_interaction.RFInteractionResult

# HHI
hhi = _hhi.hhi

# Normal test
normal_test = _normal_test.normal_test
NormalTestResult = _normal_test.NormalTestResult

# Confusion matrix
confusion_matrix = _confusion_matrix.confusion_matrix
ConfusionMatrixResult = _confusion_matrix.ConfusionMatrixResult

# U-shape
ush_test = _u_shape.ush_test
UShapeTestResult = _u_shape.UShapeTestResult

# K-fold indices
kfold_idx = _kfold_indices.kfold_idx
FoldIndices = _kfold_indices.FoldIndices

# Fairness validation
fairness_vld = _fairness_vld.fairness_vld
FairnessResult = _fairness_vld.FairnessResult

# Decision tree
decision_tree = _decision_tree.decision_tree
DecisionTreeResult = _decision_tree.DecisionTreeResult

# Stepwise forward with restrictions
step_fwdr = _step_fwdr.step_fwdr
StepFWDrResult = _step_fwdr.StepFWDrResult

# Stepwise RPC with restrictions
step_rpcr = _step_rpcr.step_rpcr
StepRPCrResult = _step_rpcr.StepRPCrResult

# Helper functions
num_slice = _helpers.num_slice
cat_slice = _helpers.cat_slice
encode_woe = _helpers.encode_woe

# Data
load_loans = _data.load_loans
get_loans_description = _data.get_loans_description

__all__ = [
    # Constants
    "DEFAULT_SPECIAL_CASES",
    "DEFAULT_SC_THRESHOLD",
    "DEFAULT_MIN_PCT_OBS",
    "DEFAULT_MIN_AVG_RATE",
    "DEFAULT_P_VALUE",
    "FLOAT_TOLERANCE",
    "DEFAULT_SC_REPLACEMENT",
    # Univariate
    "univariate",
    "imp_sc",
    "imp_outliers",
    # Bivariate
    "bivariate",
    "woe_tbl",
    "auc_model",
    "replace_woe",
    # Binning
    "cat_bin",
    # Clustering
    "rf_clustering",
    # Stepwise MIV
    "step_miv",
    "StepMIVResult",
    # Cross-validation and bootstrap
    "kfold_vld",
    "boots_vld",
    "ValidationResult",
    # Segment validation
    "segment_vld",
    "SegmentValidationResult",
    # Scaled score
    "scaled_score",
    "score_to_prob",
    # Calibration
    "rs_calibration",
    "CalibrationResult",
    # Heterogeneity
    "heterogeneity",
    # Homogeneity
    "homogeneity",
    # Predictive power
    "pp_testing",
    # Power of predictive power tests
    "power",
    "PowerResult",
    # Discriminatory power
    "dp_testing",
    # PSI
    "psi",
    "PSIResult",
    # Partitions
    "create_partitions",
    "PartitionsResult",
    # EVRS
    "evrs",
    "EVRSResult",
    # Interaction transformer
    "interaction_transformer",
    "InteractionResult",
    # Forward stepwise
    "step_fwd",
    "StepFWDResult",
    # Stepwise with risk profile check
    "step_rpc",
    "StepRPCResult",
    # Staged blocks
    "staged_blocks",
    "StagedBlocksResult",
    # Embedded blocks
    "embedded_blocks",
    "EmbeddedBlocksResult",
    # Ensemble blocks
    "ensemble_blocks",
    "EnsembleBlocksResult",
    # Near-zero variance
    "nzv",
    # SMOTE
    "smote",
    # Constrained logit
    "constrained_logit",
    "ConstrainedLogitResult",
    # RF Interaction transformer
    "rf_interaction_transformer",
    "RFInteractionResult",
    # HHI
    "hhi",
    # Normal test
    "normal_test",
    "NormalTestResult",
    # Confusion matrix
    "confusion_matrix",
    "ConfusionMatrixResult",
    # U-shape
    "ush_test",
    "UShapeTestResult",
    # K-fold indices
    "kfold_idx",
    "FoldIndices",
    # Fairness validation
    "fairness_vld",
    "FairnessResult",
    # Decision tree
    "decision_tree",
    "DecisionTreeResult",
    # Stepwise forward with restrictions
    "step_fwdr",
    "StepFWDrResult",
    # Stepwise RPC with restrictions
    "step_rpcr",
    "StepRPCrResult",
    # Helper functions
    "num_slice",
    "cat_slice",
    "encode_woe",
    # Data
    "load_loans",
    "get_loans_description",
]

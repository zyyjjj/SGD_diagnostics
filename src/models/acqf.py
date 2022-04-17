from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition import PosteriorMean
from botorch.optim import optimize_acqf
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient




def get_mfkg(model, bounds, cost_aware_utility, project):
    
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=3,
        columns=[2],
        values=[1],
    )
    
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:,:-1],
        q=1,
        num_restarts=10,
        raw_samples=1024,
        options={"batch_limit": 10, "maxiter": 200},
    )
        
    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=128,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )
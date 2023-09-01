import numpy as np
import torch
import rbo

# Weighted Mean Square Error
def weighted_mse_loss(pred, target, weight=None):
	if weight is None:
		return torch.square(torch.sub(pred, target))
	else:
		return torch.mul(weight, torch.square(torch.sub(pred, target)))

# Weighted Root Mean Square Error
def weighted_rmse(pred, target, weight=None):
	return torch.sqrt(weighted_mse_loss(pred, target, weight))

# Weighted Mean Absolute Error
def weighted_mean_absolute_error_loss(pred, target, weight=None):
	if weight is None:
		return torch.abs(torch.sub(pred, target))
	else:
		return torch.mul(weight, torch.abs(torch.sub(pred, target)))

# Pearson Correlation Coefficient
def pearson_correlation_coefficient(pred, target):
	pred_mean = torch.mean(pred)
	target_mean = torch.mean(target)

	p = torch.sum(torch.mul(torch.sub(pred, pred_mean), torch.sub(target, target_mean)))
	d = torch.mul(torch.sqrt(torch.sum(torch.square(torch.sub(pred, pred_mean)))), torch.sqrt(torch.sum(torch.square(torch.sub(target, target_mean)))))
	return torch.div(p,d)


"""
Custom penalization - according to a certain threshold, prediction above is considered as "of high interest", and we penalize 
predictions below when the ground truth is above to make our model relunctant to predict false negative
"""
def penalize_false_negative(raw_loss, pred, target, threshold=4, penalization=3):
	pred_below = pred < threshold
	target_above = target > threshold
	return torch.where(pred_below & target_above, torch.mul(raw_loss, penalization), raw_loss)


"""
Rank-Biased Overlap (RBO): Similarity between nonconjoint or incomplete rankings
N.B.: We use the RBO package built by RBO's paper authors
"""
class RBOLoss(torch.nn.Module):
	def __init__(self, reduction='none'):
		if reduction not in ['mean', 'sum', 'none']:
			raise ValueError('RBO Loss: Reduction `{}` not implemented'.format(reduction))
		self.reduction = reduction
		super(RBOLoss, self).__init__()

	# We expect to have, for each node, two array `prediction` and `target` of the same shape containing respectively the predicted output and the ground truth for each of the edges coming out of this node
	def calculate_rbo(self, prediction, target):
		# Rank GT and Predictions
		gt_ranking = torch.argsort(target, descending=True, dim=0).flatten()
		pred_ranking = torch.argsort(prediction, descending=True, dim=0).flatten()
		# RBO package is not accepting tensors
		return (rbo.RankingSimilarity(gt_ranking.numpy(), pred_ranking.numpy()).rbo(), gt_ranking.size(dim=0))

	def forward(self, predictions, targets):
		rbos, weights = torch.ones(len(predictions), dtype=float), torch.zeros(len(targets), dtype=float)

		for idx, (pred, gt) in enumerate(zip(predictions, targets)):
			rbo, weight = self.calculate_rbo(pred.cpu(), gt.cpu())
			rbos[idx] = 1 - rbo # Computing loss, RBO=1 is perfect prediction
			weights[idx] = weight

		# Attempt to use RBO as trainable loss - Not functional yet!
		rbos.requires_grad = True
		weights.requires_grad = True

		loss = torch.mul(rbos, weights)
		if self.reduction == 'mean':
			loss = torch.div(loss.sum(), weights.sum())
		elif self.reduction == 'sum':
			loss = loss.sum()
		return loss
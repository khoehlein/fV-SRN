from torch import Tensor, nn


class IFieldEvaluator(nn.Module):
    """
    Baseclass for everything that accepts position samples in d_in-dimensionalspace
    and returns d_out-dimensional field output
    """

    def __init__(self, in_dimension, out_dimension, device):
        super(IFieldEvaluator, self).__init__()
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.device = device

    def evaluate(self, positions: Tensor) -> Tensor:
        """
        Interface function for calling the field evaluation
        """
        positions = self._verify_positions(positions)
        out = self.forward(positions)
        return self._verify_outputs(out)

    def _verify_positions(self, positions: Tensor):
        assert len(positions.shape) == 2 and positions.shape[-1] == self.in_dimension
        if self.device is None or positions.device == self.device:
            return positions
        return positions.to(self.device)

    def _verify_outputs(self, outputs: Tensor) -> Tensor:
        if len(outputs.shape) < 2:
            outputs = outputs[:, None]
        assert len(outputs.shape) == 2 and outputs.shape[-1] == self.out_dimension
        return outputs

    def forward(self, positions: Tensor) -> Tensor:
        """
        Function to implement the field evaluation logic
        """
        raise NotImplementedError()

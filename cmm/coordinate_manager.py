import torch
from neighbor_list import CellList

class CoordinateManager:
    def __init__(self, positions: torch.Tensor, box: torch.Tensor):
        self.box = box.clone()
        self.box_inv = torch.linalg.inv(box)
        self.positions = positions.clone()

    def wrap_coordinates(positions: torch.Tensor, box: torch.Tensor):
        raise NotImplementedError

    def applyPBC3D(self, drVecs: torch.Tensor):
        """
        Apply periodic boundary conditions to a set of vectors

        Parameters
        ----------
        drVecs: torch.Tensor
            Real space vectors in Cartesian, with shape (N, 3)
        box: torch.Tensor
            Simulation box, with axes arranged in rows, shape (3, 3)
        boxInv: torch.Tensor
            Inverse of the simulation box matrix, with axes arranged in rows, shape (3, 3)
        """
        # NOTE(JOE): In the future, we should support having PBCs turned off for certain
        # directions as well as completely turned off. I named this applyPBC3D so that we
        # can implement applyPBC2D and applyPBC1D then switch on the appropriate function
        # call with an enum which is determined when we set up the simulation.
        dsVecs = torch.matmul(drVecs, self.box_inv)
        dsVecsPBC = dsVecs - torch.floor(dsVecs + 0.5)
        drVecsPBC = torch.matmul(dsVecsPBC, self.box)
        return drVecsPBC

if __name__ == "__main__":
    grid = torch.linspace(-10.0, 10.0, 10)
    positions = torch.cartesian_prod(grid, grid, grid)
    box = torch.eye(3) * 20.0
    A = torch.rand(3, 3)
    coordinate_manager = CoordinateManager(positions, box)
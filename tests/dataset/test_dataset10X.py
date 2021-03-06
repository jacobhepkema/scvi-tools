import scvi
import tarfile
import os
import scanpy as sc
import pytest
from .utils import unsupervised_training_one_epoch


def test_populate_and_train_one_v1(save_path):
    sp = os.path.join(save_path, "10X")
    dataset = scvi.data.dataset_10x(
        dataset_name="cd4_t_helper",
        remove_extracted_data=True,
        save_path=sp,
    )
    scvi.data.setup_anndata(dataset)
    unsupervised_training_one_epoch(dataset)


def test_brain_small(save_path):
    sp = os.path.join(save_path, "10X")
    dataset = scvi.data.dataset_10x(
        dataset_name="neuron_9k",
        save_path=sp,
        remove_extracted_data=True,
    )
    scvi.data.setup_anndata(dataset)
    unsupervised_training_one_epoch(dataset)


def test_pbmc_cite(save_path):
    file_path = os.path.join(
        save_path, "10X/pbmc_10k_protein_v3/filtered_feature_bc_matrix.tar.gz"
    )
    sp = os.path.join(save_path, "10X/pbmc_10k_protein_v3/")
    tar = tarfile.open(file_path, "r:gz")
    tar.extractall(path=sp)
    tar.close()
    dataset = sc.read_10x_mtx(
        os.path.join(sp, "filtered_feature_bc_matrix"), gex_only=False
    )
    scvi.data.organize_cite_seq_10x(dataset)
    scvi.data.setup_anndata(dataset, protein_expression_obsm_key="protein_expression")
    unsupervised_training_one_epoch(dataset)


@pytest.mark.internet
def test_download_dataset_10x(save_path):
    scvi.data.dataset_10x("hgmm_1k_v3", save_path=save_path)

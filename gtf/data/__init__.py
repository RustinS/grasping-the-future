def __getattr__(name: str):
    """Lazy imports to avoid double-import warning when running compute_stats as __main__."""
    if name == "load_stats":
        from gtf.data.compute_stats import load_stats

        return load_stats
    if name in ("GTFDataset", "gtf_collate"):
        from gtf.data.dataset import GTFDataset, gtf_collate

        return {"GTFDataset": GTFDataset, "gtf_collate": gtf_collate}[name]
    raise AttributeError(f"module 'gtf.data' has no attribute {name!r}")

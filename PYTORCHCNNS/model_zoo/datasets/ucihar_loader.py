import torchhd.datasets
from torch.utils.data import DataLoader, Subset

def get_activity_loader(activity, batch_size=64, train=True):
    """
    Creates a DataLoader for a specific activity from the UCI HAR dataset.

    Args:
        activity (str): The activity to filter. Should be one of:
                        "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", 
                        "SITTING", "STANDING", or "LAYING".
        batch_size (int): The batch size for the DataLoader.
        train (bool): Whether to load training data (True) or test data (False).

    Returns:
        DataLoader: A PyTorch DataLoader containing only the samples for the specified activity.
    """
    
    activity = activity.upper()
    activity_map = {
        "WALKING": 0,
        "WALKING_UPSTAIRS": 1,
        "WALKING_DOWNSTAIRS": 2,
        "SITTING": 3,
        "STANDING": 4,
        "LAYING": 5
    }
    
    if activity not in activity_map:
        raise ValueError(f"Activity {activity} is not recognized. Must be one of {list(activity_map.keys())}.")
    
    activity_index = activity_map[activity]
    
    # Set the root directory for UCI HAR dataset.
    root_dir = "./datasets/ucihar"
    dataset = torchhd.datasets.UCIHAR(root=root_dir, train=train, download=True)
    
    # Filter the dataset to only include samples with the desired activity label.
    indices = [i for i, (_, label) in enumerate(dataset) if label.item() == activity_index]
    activity_subset = Subset(dataset, indices)
    
    loader = DataLoader(activity_subset, batch_size=batch_size, shuffle=True)
    
    return loader

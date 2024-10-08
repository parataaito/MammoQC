# Downloading the VINDR Mammography Dataset

This README provides instructions on how to download the VINDR Mammography Dataset.

## Dataset Information

The VINDR Mammography Dataset is a large-scale mammography dataset for breast cancer detection. For more information about the dataset, visit the official VINDR page:

[VINDR Mammography Dataset](https://vindr.ai/datasets/mammo)

## Download Instructions

### Step 1: Register on PhysioNet

The dataset is hosted on PhysioNet. You need to register and get approved before you can access the files.

1. Go to the PhysioNet page for the VINDR Mammography Dataset:
   [https://www.physionet.org/content/vindr-mammo/1.0.0/](https://www.physionet.org/content/vindr-mammo/1.0.0/)

2. Click on "Register" or "Sign In" if you already have an account.

3. Follow the registration process and wait for approval.

### Step 2: Access the Files

Once your registration is approved and you're signed in:

1. Navigate back to the VINDR Mammography Dataset page on PhysioNet.
2. The "Files" section should now be available to you.

### Step 3: Download the Dataset

You have two options for downloading the dataset:

#### Option 1: Manual Download

In the "Files" section, you can manually click and download the ZIP file.

#### Option 2: Command Line Download

You can use the following `wget` command to download the entire dataset:

```bash
wget -r -N -c -np --user your_username --ask-password https://physionet.org/files/vindr-mammo/1.0.0/
```

If you are on Windows, you may need to use WSL to make it wortk.

Replace `your_username` with your PhysioNet username. You will be prompted to enter your PhysioNet password.

This command does the following:
- `-r`: recursively download all files
- `-N`: only download newer files
- `-c`: continue partially downloaded files
- `-np`: don't ascend to parent directory
- `--user`: specify your PhysioNet username
- `--ask-password`: prompt for your password

**Note**: Ensure you have sufficient storage space before downloading the entire dataset.

## Support

If you encounter any issues during the download process, please refer to the PhysioNet support or contact the VINDR dataset maintainers.

Happy downloading!
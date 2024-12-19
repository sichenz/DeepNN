# DeepNN

DeepNN is a deep learning framework designed for Professor Xiao Liu's class. This guide provides step-by-step instructions to help you set up and run DeepNN on your local machine.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
  - [1. Install Anaconda](#1-install-anaconda)
  - [2. Create and Activate a Virtual Environment](#2-create-and-activate-a-virtual-environment)
  - [3. Install R Data.Table Package](#3-install-r-datatable-package)
- [Clone the Repository](#clone-the-repository)
- [Install Necessary Python Packages](#install-necessary-python-packages)
- [Run the Pipeline](#run-the-pipeline)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- [Anaconda](https://www.anaconda.com/products/distribution) (for managing environments)
- [Git](https://git-scm.com/downloads) (for cloning the repository)
- [Python 3.8](https://www.python.org/downloads/release/python-380/)
- [R](https://www.r-project.org/) (required for the `r-data.table` package)

## Setup Instructions

Follow these steps to set up DeepNN on your machine.

### Create and Activate a Virtual Environment

1. **Open your terminal or command prompt.**

2. **Create a new Conda environment named `deepnn` with Python 3.8:**

   ```bash
   conda create --name deepnn python=3.8
   ```

3. **Activate the `deepnn` environment:**

   ```bash
   conda activate deepnn
   ```

4. **Install `r-data.table` using Conda:**

   ```bash
   conda install -c conda-forge r-data.table
   ```


## Clone the Repository

1. **Navigate to the directory where you want to clone the repository:**

   ```bash
   cd /path/to/your/preferred/directory
   ```

2. **Clone the DeepNN repository:**

   ```bash
   git clone https://github.com/sichenz/DeepNN.git
   ```

3. **Navigate into the cloned repository:**

   ```bash
   cd DeepNN
   ```

## Install Necessary Python Packages

1. **Upgrade `pip`, `setuptools`, and `wheel`:**

   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. **Run the setup script to install additional dependencies:**

   ```bash
   bash setup/run_setup.sh
   ```

   - **Note:** Ensure that the `run_setup.sh` script has execute permissions. If not, you can make it executable with:

     ```bash
     chmod +x setup/run_setup.sh
     ```

## Run the Pipeline

1. **Open the `pipeline.sh` script in a text editor or on VSCode**

2. **Locate the `cd` command in the script and update it to point to the absolute path of your cloned `DeepNN` repository.**

   - **Example:**

     ```bash
     cd /Users/yourusername/path/to/DeepNN
     ```

3. **Save and close the `pipeline.sh` script.**

4. **Make sure the `pipeline.sh` script has execute permissions:**

   ```bash
   chmod +x pipeline.sh
   ```

5. **Run the pipeline script:**

   ```bash
   bash pipeline.sh
   ```

## Re-run the Program

If you want to rerun the DeepNN program, you need to remove specific directories to reset the environment. Use the following commands:
   ```bash
   rm -rf ~/dnn-paper/final_000
   rm -rf ~/dnn-paper/final_001
   rm -rf ~/dnn-paper/final_002
   rm -rf ~/dnn-paper/paper
   `````
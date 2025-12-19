# Detailed Setup Guide for SHARP - Beginner's Edition

Welcome! This guide will walk you through setting up the SHARP 3D Prediction project on your computer, step by step. Don't worry if you're new to programming or command-line tools—we'll explain everything you need to know.

## What is SHARP?

SHARP is a tool that can take a single photograph and create a 3D representation of it, allowing you to view it from different angles. This project includes a user-friendly web interface where you can upload images and download the 3D results.

## Table of Contents

1. [Prerequisites - What You Need](#prerequisites---what-you-need)
2. [Step 1: Install Conda](#step-1-install-conda)
3. [Step 2: Download the Project](#step-2-download-the-project)
4. [Step 3: Set Up the Project](#step-3-set-up-the-project)
5. [Step 4: Start the Web Interface](#step-4-start-the-web-interface)
6. [Troubleshooting](#troubleshooting)
7. [Alternative Setup Methods](#alternative-setup-methods)

---

## Prerequisites - What You Need

Before we begin, here's what you'll need:

### Hardware Requirements
- A computer running **macOS** (for Windows or Linux, see [Alternative Setup Methods](#alternative-setup-methods))
- At least **8 GB of RAM** (16 GB or more recommended)
- At least **5 GB of free disk space**
- **Internet connection** for downloading software and dependencies

### What You Don't Need
- You don't need to know how to code
- You don't need prior experience with Python or machine learning
- You don't need to understand how the AI works (but you can learn if you want!)

---

## Step 1: Install Conda

Conda is a package manager that helps organize Python and its libraries. Think of it as an app store for Python tools.

### What is Conda?

Conda creates isolated "environments" for different projects, so they don't interfere with each other. We'll use it to install Python and all the tools SHARP needs.

### Download and Install Miniconda

1. **Go to the Miniconda download page:**
   - Open your web browser and visit: https://docs.conda.io/en/latest/miniconda.html

2. **Download the macOS installer:**
   - Look for the **macOS** section
   - Download the **latest Python 3.x** installer for your Mac:
     - If you have an **M1/M2/M3 Mac** (Apple Silicon): Choose the `Apple M1` or `arm64` version
     - If you have an **Intel Mac**: Choose the `Intel x86_64` version
   - The file will be named something like `Miniconda3-latest-MacOSX-arm64.pkg` or `Miniconda3-latest-MacOSX-x86_64.pkg`

3. **Install Miniconda:**
   - Double-click the downloaded `.pkg` file
   - Follow the installation wizard:
     - Click "Continue" through the introduction screens
     - Accept the license agreement
     - Choose "Install for me only" (recommended)
     - Click "Install" and enter your password when prompted
   - When installation completes, click "Close"

4. **Verify the installation:**
   - Open **Terminal** (you can find it in Applications → Utilities → Terminal)
   - Type the following command and press Enter:
     ```bash
     conda --version
     ```
   - You should see something like `conda X.X.X` (the exact version number may vary)
   - **If you get an error** saying "conda: command not found":
     - Close Terminal completely and open it again
     - Try the command again
     - If it still doesn't work, see [Troubleshooting](#troubleshooting)

---

## Step 2: Download the Project

Now we need to download the SHARP project files to your computer.

### Option A: Download via GitHub (Easiest for Beginners)

1. **Go to the GitHub repository:**
   - Visit the repository URL where this project is hosted (check the address bar or the project README for the correct URL)

2. **Download the ZIP file:**
   - Click the green **"Code"** button
   - Click **"Download ZIP"**
   - The file will be saved to your Downloads folder

3. **Extract the ZIP file:**
   - Go to your Downloads folder
   - Double-click the `ml-sharp-main.zip` (or similar name) file
   - macOS will automatically extract the folder

4. **Move to a convenient location (optional but recommended):**
   - Create a folder in your Documents called `Projects`
   - Drag the extracted `ml-sharp-main` folder into `Projects`
   - Rename it to just `ml-sharp` to make it simpler

### Option B: Using Git (If You Have It Installed)

If you're comfortable with Git or have it installed:

```bash
cd ~/Documents
git clone <repository-url>
cd ml-sharp
```

(Replace `<repository-url>` with the actual Git URL from the GitHub repository)

---

## Step 3: Set Up the Project

Now we'll set up the Python environment and install all the required libraries.

### Using Terminal

1. **Open Terminal** (Applications → Utilities → Terminal)

2. **Navigate to the project folder:**
   - Type `cd` followed by a space
   - Drag the `ml-sharp` folder from Finder onto the Terminal window
   - Press Enter
   - Your command should look something like: `cd /Users/YourName/Documents/Projects/ml-sharp`

3. **Initialize conda in your terminal:**
   ```bash
   conda init bash
   ```
   - Close Terminal and open it again

4. **Create the Python environment:**
   ```bash
   conda create -n sharp python=3.13 -y
   ```
   - This creates a special environment named "sharp" with Python 3.13
   - The process may take a few minutes
   - Wait for it to complete

5. **Activate the environment:**
   ```bash
   conda activate sharp
   ```
   - You should see `(sharp)` appear at the beginning of your command prompt

6. **Install the main project dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   - This installs all the machine learning libraries SHARP needs
   - **This will take 5-15 minutes** depending on your internet speed
   - You'll see a lot of text scrolling by—this is normal
   - Be patient and let it finish

7. **Install the web interface dependencies:**
   ```bash
   pip install -r src/sharp/web/requirements.txt
   ```
   - This installs the web server components
   - This should be faster, taking about 1-2 minutes

8. **Verify the installation:**
   ```bash
   sharp --help
   ```
   - If successful, you'll see the SHARP help menu
   - This means everything is installed correctly!

---

## Step 4: Start the Web Interface

Now for the exciting part—starting the web interface!

### If start.command Works (Try This First)

1. **In Finder, navigate to the ml-sharp folder**

2. **Double-click the `start.command` file**

3. **If macOS blocks it with a security warning:**
   - Right-click (or Control-click) on `start.command`
   - Select **"Open"** from the menu
   - Click **"Open"** in the dialog that appears
   - macOS will remember your choice for this file
   
   OR
   
   - Go to **System Settings → Privacy & Security**
   - Scroll down to find a message about `start.command` being blocked
   - Click **"Open Anyway"**
   - Right-click on `start.command` again and choose **"Open"**

4. **A Terminal window will open** with the SHARP logo

5. **The script will automatically:**
   - Check your conda installation
   - Create or activate the environment
   - Install any missing dependencies
   - Start the web server

6. **When you see "Starting Sharp Web Interface":**
   - Open your web browser (Safari, Chrome, Firefox, etc.)
   - Go to: **http://localhost:8000**
   - You should see the SHARP web interface!

### If start.command Doesn't Work (Manual Method)

If the automatic script doesn't work, don't worry! Here's how to start it manually:

1. **Open Terminal**

2. **Navigate to the project folder:**
   ```bash
   cd /path/to/ml-sharp
   ```
   (Replace with your actual path, or drag the folder onto Terminal)

3. **Activate the conda environment:**
   ```bash
   conda activate sharp
   ```

4. **Start the web server:**
   ```bash
   python src/sharp/web/app.py
   ```

5. **Open your web browser and go to:**
   - **http://localhost:8000**

6. **To stop the server:**
   - Press **Control + C** in the Terminal window

---

## Troubleshooting

### "conda: command not found"

**Problem:** Terminal doesn't recognize the `conda` command.

**Solutions:**

1. **Initialize conda:**
   ```bash
   ~/miniconda3/bin/conda init bash
   ```
   Then close and reopen Terminal.

2. **Add conda to your PATH manually:**
   ```bash
   echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bash_profile
   source ~/.bash_profile
   ```

3. **If you installed Anaconda instead of Miniconda:**
   ```bash
   echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bash_profile
   source ~/.bash_profile
   ```

### "Permission denied" when running start.command

**Problem:** The script doesn't have permission to run.

**Solution:**

1. Open Terminal
2. Navigate to the project folder
3. Make the script executable:
   ```bash
   chmod +x start.command
   ```
4. Try double-clicking it again

### "Failed to create conda environment"

**Problem:** Error creating the Python environment.

**Solutions:**

1. **Make sure you have enough disk space** (at least 5 GB free)

2. **Update conda:**
   ```bash
   conda update conda
   ```

3. **Try creating the environment with a different Python version:**
   ```bash
   conda create -n sharp python=3.11 -y
   ```

### Installation is Taking Forever / Stuck

**Problem:** `pip install` seems frozen or very slow.

**Solutions:**

1. **Be patient:** The first installation can take 10-20 minutes, especially for PyTorch
2. **Check your internet connection**
3. **If truly stuck (no progress for 30+ minutes):**
   - Press Control + C to cancel
   - Try again:
     ```bash
     pip install -r requirements.txt --no-cache-dir
     ```

### "Port 8000 is already in use"

**Problem:** Another application is using port 8000.

**Solution:**

1. **Find and stop the process using port 8000:**
   ```bash
   lsof -ti:8000 | xargs kill -9
   ```

2. **Or modify the port in the app.py file:**
   - Open `src/sharp/web/app.py` in a text editor
   - Look for the line near the end: `uvicorn.run(app, host="0.0.0.0", port=8000)`
     (Tip: Search for `port=` to find it quickly)
   - Change `port=8000` to `port=8080` (or another available port number)
   - Save the file and start the server again
   - Then access it at: http://localhost:8080 (or your chosen port)

### The Web Interface Won't Load

**Problem:** Browser shows an error when accessing http://localhost:8000

**Solutions:**

1. **Make sure the server is actually running** (check Terminal for errors)
2. **Try a different browser**
3. **Clear your browser cache**
4. **Check if you're using the correct URL:** http://localhost:8000 (not https)
5. **Look for error messages in the Terminal** and search for them online or see below

### "ModuleNotFoundError: No module named 'sharp'"

**Problem:** Python can't find the SHARP module.

**Solution:**

1. Make sure you're in the correct directory
2. Make sure the conda environment is activated (you should see `(sharp)` in the prompt)
3. Try installing again:
   ```bash
   pip install -r requirements.txt
   ```

### Out of Memory Errors

**Problem:** Your computer runs out of RAM.

**Solutions:**

1. **Close other applications** to free up memory
2. **Restart your computer** and try again
3. **Consider using the CLI** instead of the web interface for large batches of images

---

## Alternative Setup Methods

### For Windows Users

The `start.command` script is macOS-specific, but you can follow these steps on Windows:

1. **Install Miniconda for Windows:**
   - Download from: https://docs.conda.io/en/latest/miniconda.html
   - Choose the Windows installer

2. **Open Anaconda Prompt** (search for it in the Start menu)

3. **Follow steps 3 and 4** from above, using the same commands

### For Linux Users

The setup is very similar to macOS:

1. **Install Miniconda for Linux:**
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

2. **Follow the macOS instructions** using Terminal

### Using the Command Line Interface (CLI)

If you prefer not to use the web interface, you can use SHARP directly from the command line:

1. **Activate the environment:**
   ```bash
   conda activate sharp
   ```

2. **Run a prediction:**
   ```bash
   sharp predict -i /path/to/input/images -o /path/to/output/gaussians
   ```

3. **See all options:**
   ```bash
   sharp --help
   ```

---

## What's Next?

Once you have the web interface running:

1. **Upload an image** using the web interface
2. **Wait for processing** (usually takes a few seconds per image)
3. **Download the 3D Gaussian file** (.ply format)
4. **View it** using a 3D Gaussian viewer (the web interface may include a viewer)

For more advanced usage, check out the [README.md](README.md) file in the project folder.

---

## Getting Help

If you're still stuck after trying these solutions:

1. **Check the project's GitHub Issues page** to see if others have had similar problems
2. **Read the main [README.md](README.md)** for additional technical details
3. **Create a new GitHub Issue** describing your problem:
   - Include your operating system version
   - Include any error messages you see
   - Describe what you've already tried

---

## Tips for Success

- **Be patient:** The first setup takes time, but subsequent runs will be much faster
- **Read error messages:** They often tell you exactly what's wrong
- **Google is your friend:** Copy error messages and search for them
- **Keep your terminal open:** Don't close Terminal while the server is running
- **Save your work:** The web interface processes images but doesn't permanently store them

---

## Summary of Commands

Here's a quick reference of the key commands:

```bash
# Navigate to project
cd /path/to/ml-sharp

# Activate environment
conda activate sharp

# Start web interface (manual method)
python src/sharp/web/app.py

# Use CLI
sharp predict -i /path/to/images -o /path/to/output

# Deactivate environment when done
conda deactivate
```

---

**Congratulations!** 🎉 You now have SHARP set up and running. Enjoy creating 3D representations from your photos!

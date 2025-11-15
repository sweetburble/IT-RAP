# MAAD-Face Dataset Auto Downloader & Setup

## 1. Create target directory
```bash
mkdir -p MAAD-Face
cd MAAD-Face
```

## 2. Download entire Google Drive folder using gdown (folder ID extracted from link)
[MAAD-Face dataset w/ attribute csv file](https://drive.google.com/drive/folders/1iRbo_IwdQZnLJ3U15oUVMvd5Ab8oFADV?usp=drive_link)


## 3. Move downloaded files to current directory (Google Drive folder name may vary)

## 4. Move CSV files to current directory root (if inside folder)

## 5. Unzip archive.zip
```bash
unzip archive.zip
```

## 6. Identify extracted folder and rename to "data"
```bash
UNZIP_FOLDER=$(ls -d */ | grep -v "$DL_FOLDER" | grep -v "data" | sed 's#/##')
mv "$UNZIP_FOLDER" data
```

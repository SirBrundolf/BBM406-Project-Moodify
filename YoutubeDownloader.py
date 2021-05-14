from __future__ import unicode_literals
import youtube_dl


def read_file(file_name):
    file = open(file_name, "r", encoding="utf8")
    comments = file.readlines()
    return comments


youtube_base = "https://www.youtube.com/watch?v="

count = 0
files = read_file("unbalanced_train_segments.csv")
for i in files:
    if count > 100:  # Setting a limit for each label to split the data equally
        break
    data = i.split(", ")
    tags = data[3].split(",")
    if tags[-1] == 'Tender"\n':  # Download only the music with the given label, to split the data
        ydl_opts = {
            'ignoreerrors': True,
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '240',
            }],
            'postprocessor_args': [
                '-ss', '00:00:' + str(int(float(data[1]))),
                '-t', '00:00:10',
            ],
        }
        link = youtube_base + data[0]
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])
        count += 1
    print("YEP")

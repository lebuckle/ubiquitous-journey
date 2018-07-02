#!/bin/bash

video_1=$1
video_2=$2
folder_loc=$3

help_message="Usage: ./concat_videos video1 video2"

# Check the variables are set
if [ -z ${video_1} ]; 
	then echo "$help_message"; 
	else echo "video one is '$video_1'"; 
fi

mencoder -oac copy -ovc copy video_1 video_2 -o full_movie.avi
# Download files (the zip file is split into two files) and then unzip
wget https://github.com/lukemelas/projection-conditioned-point-cloud-diffusion/releases/download/v0.0/example-vis-parts.zip
wget https://github.com/lukemelas/projection-conditioned-point-cloud-diffusion/releases/download/v0.0/example-vis-parts.z01
zip -FF example-vis-parts.zip --out example-vis.zip && rm example-vis-parts.zip example-vis-parts.z01
unzip example-vis.zip && rm example-vis.zip
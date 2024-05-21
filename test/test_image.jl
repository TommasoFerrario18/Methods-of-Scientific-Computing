include("../src/Utils.jl")
include("../src/Dct2.jl")

using .Dct2 
using FFTW
using .Utils


img= Utils.LoadBmpImage("input.bmp")

#print("Insert F: ")
F = 50#readline()  

#print("Insert d(2F-2): ")
d = 50#readline()  

# println(size(Dct2.ResizeMatrix(input2, F)))
# println(Dct2.Compress(input2, F))

out = Dct2.ApplyDct2OnImage(img, F, d)

Utils.SaveBmpImage(out, "output.bmp")
#println(Dct2.ApplyDct2OnImage(input2, F, d))

# out = Dct2.ApplyDct2OnImage(input2, F, d)


# println(out)
# exit()
# Dct2.SaveBmpImage(out, "output.bmp")
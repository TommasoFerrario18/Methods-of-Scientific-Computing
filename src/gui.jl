include("../src/Utils.jl")
include("../src/Dct2.jl")

using PlotlyJS
using Dash
using .Utils
using .Dct2
using Base64
using Images
using ImageIO
using FileIO
using Dates

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash(external_stylesheets=external_stylesheets)

# Define the layout of the app
app.layout = html_div() do
    html_h1("Progetto Metodi Calcolo Scientifico"),
    dcc_upload(
        id="upload-image",
        children=html_div([
            "Drag and Drop or ",
            html_a("Select Files")
        ]),
        style=Dict(
            "width" => "100%",
            "height" => "60px",
            "lineHeight" => "60px",
            "borderWidth" => "1px",
            "borderStyle" => "dashed",
            "borderRadius" => "5px",
            "textAlign" => "center",
            "margin" => "10px"
        ),
        multiple=false
    ),
    html_div(
        id="input-size-block",
        children=[
            html_h5("Size of the block\t", style=Dict("display" => "inline-block")),
            dcc_input(id="size-block", type="number", placeholder="Size of the block", value=1, min=1, max=100, step=1, style=Dict("display" => "inline-block", "margin" => "10px"))
        ],
        style=Dict("display" => "inline-block", "width" => "100vg", "padding" => "10px")),
    html_div(id="output-image-upload"),
    html_div(id="processing-image-upload")
end

function parse_contents_fig(contents, filename, date)
    return html_div([
        html_h5(filename),
        html_h6(date),
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html_img(src=contents, style=Dict("height" => "50%", "width" => "50%")),
        html_hr(),
        html_div("Raw Content"),
        html_pre(string(first(contents, 100), "..."), style=Dict(
            "whiteSpace" => "pre-wrap",
            "wordBreak" => "break-all"
        ))
    ])
end

callback!(
    app,
    Output("output-image-upload", "children"),
    Input("upload-image", "contents"),
    State("upload-image", "filename"),
    State("upload-image", "last_modified"),
) do contents, filename, last_modified
    if !(contents isa Nothing)
        children = [
            parse_contents_fig(c...) for c in
            zip([contents], filename, last_modified)]
        return children
    end
end


callback!(
    app,
    Output("processing-image-upload", "children"),
    Input("size-block", "value"),
    Input("upload-image", "contents"),
    State("upload-image", "filename")
) do size_block, contents, filename
    if !(contents isa Nothing)
        println("Processing image")
        
        open("code.txt", "w") do f
            write(f, contents)
        end
        contents_img = split(contents, ",")

        img_data = Base64.base64decode(contents_img[2])
        println("Image data: ", typeof(img_data))
        temp_filename = "temp_image.bmp"
        open(temp_filename, "w") do f
            write(f, img_data)
        end

        # Load the image
        img = Utils.LoadBmpImage(temp_filename)
        println("Image loaded: ", typeof(img))
        # Apply the DCT2
        out = Dct2.ApplyDct2OnImage(img, size_block, 50)
        # Save the image
        Utils.SaveBmpImage(out, "output.bmp")
        println("Image saved")
        
        output_img = Utils.LoadBmpImage("output.bmp") 
        img_path = "output.bmp"
        img = open(img_path) do file
            read(file, String)  
        end
        output_code = base64encode(img)
        
        return parse_contents_fig("data:image/bmp;base64,$(output_code)", "output.bmp", now())
    end
end

# Run the app
run_server(app, "0.0.0.0", debug=true)
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
        multiple=true
    ),
    html_div(
        id="input-size-block",
        children=[
            html_h5("Size of the block\t", style=Dict("display" => "inline-block")),
            dcc_input(id="size-block", type="number", placeholder="Size of the block", value=1, min=1, max=100, step=1, style=Dict("display" => "inline-block", "margin" => "10px"))
        ],
        style=Dict("display" => "inline-block", "width" => "100%", "margin" => "10px")),
    html_div(id="output-image-upload"),
    html_div(id="processing-image-upload")
end

function parse_contents_fig(contents, filename, date)
    return html_div([
        html_h5(filename),
        html_h6(Libc.strftime(date)),
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html_img(src=contents, style=Dict("height" => "50%", "width" => "50%")),
        html_hr(),
        html_div("Raw Content"),
        html_pre(string(contents[1:200], "..."), style=Dict(
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
            zip(contents, filename, last_modified)]
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
        println("Contents: ", contents[1])
        println("Filename: ", contents[2])

        img_data = Base64.base64decode(contents[1])
        println("Image data: ", typeof(img_data))

        temp_filename = "temp_image.bmp"
        open(temp_filename, "w") do f
            write(f, img_data)
        end

        
        # Read the image from the file
        img = load(temp_filename)
        
        # Convert the image to a matrix
        img_matrix = channelview(img)
        
        # Print the matrix
        println(img_matrix)

        # Load the image
        # img = Utils.LoadBmpImage("input.bmp")
        # println("Image loaded: ", typeof(img))
        # # Apply the DCT2
        # out = Dct2.ApplyDct2OnImage(img, size_block, 50)
        # # Save the image
        # Utils.SaveBmpImage(out, "output.bmp")
        # # Return the image
        # return parse_contents_fig(Base64.base64encode(out), "output.bmp", 0)
    end
end

# Run the app
run_server(app, "0.0.0.0", debug=true)
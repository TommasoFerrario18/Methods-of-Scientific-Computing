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
    html_div(
        id="page",
        children=[
            html_h1("Progetto Metodi Calcolo Scientifico"),
            dcc_upload(
                id="upload-image",
                children=html_div([
                    "Drag and Drop or ",
                    html_a("Select Files")
                ]),
                style=Dict("lineHeight" => "60px",
                    "borderWidth" => "1px",
                    "borderStyle" => "dashed",
                    "borderRadius" => "5px",
                    "textAlign" => "center",
                    "margin" => "10px"
                ),
                multiple=false
            ),
            html_div(
                children=[
                    html_div(
                        id="input-size-block",
                        children=[
                            html_h5("Size of the block\t", style=Dict("display" => "inline-block")),
                            dcc_input(id="size-block", type="number", placeholder="Size of the block", value=1, min=1, max=100, step=1, style=Dict("display" => "inline-block", "margin" => "10px"))
                        ],
                        style=Dict("display" => "inline-block", "width" => "100vg", "padding" => "10px")
                    ),
                    html_div(
                        id="Compression index",
                        children=[
                            html_h5("Index of compression\t", style=Dict("display" => "inline-block")),
                            dcc_input(id="compression-index", type="number", placeholder="Compression index", value=0, min=0, step=1, style=Dict("display" => "inline-block", "margin" => "10px"))
                        ],
                        style=Dict("display" => "inline-block", "width" => "100vg", "padding" => "10px")
                    ),
                    html_button("Compress", id="compress-button", style=Dict("display" => "inline-block", "margin" => "15px"))
                ],
                style=Dict("display" => "flex", "flex-direction" => "row", "justify-content" => "center")
            ),
            html_div(
                children=[
                    html_div(
                        id="output-image-upload",
                        style=Dict("display" => "flex", "flex-direction" => "column", "justify-content" => "center", "width" => "50%")
                    ),
                    html_div(
                        id="processing-image-upload",
                        style=Dict("display" => "flex", "flex-direction" => "column", "justify-content" => "center", "width" => "50%")
                    )
                ],
                style=Dict("display" => "flex", "flex-direction" => "row", "justify-content" => "center", "max-width" => "100vw")
            )
        ],
        style=Dict("display" => "flex", "flex-direction" => "column")
    )



end

function parse_contents_fig(contents, filename, date)
    return html_div([
            html_h5(filename),
            html_h6(date),
            # HTML images accept base64 encoded strings in the same format
            # that is supplied by the upload
            html_img(src=contents, style=Dict("height" => "auto", "width" => "auto")),
            html_hr(),
            html_div("Raw Content"),
            html_pre(string(first(contents, 100), "..."), style=Dict(
                "whiteSpace" => "pre-wrap",
                "wordBreak" => "break-all"
            ))
        ],
        style=Dict("display" => "flex", "flex-direction" => "column", "justify-content" => "center", "padding" => "20px"))
end

callback!(
    app,
    Output("compression-index", "max"),
    Input("size-block", "value")
) do value
    return value * 2 - 2
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
    Input("compress-button", "n_clicks"),
    State("size-block", "value"),
    State("upload-image", "contents"),
    State("upload-image", "filename"),
    State("compression-index", "value")
) do n_clicks, size_block, contents, filename, compression_index
    if !(contents isa Nothing)
        println("Processing image")

        img_data = Base64.base64decode(split(contents, ",")[2])

        println("Save image on FS")
        temp_filename = "temp_image.bmp"
        open(temp_filename, "w") do f
            write(f, img_data)
        end

        # Load the image
        img = Utils.LoadBmpImage(temp_filename)
        # Apply the DCT2
        println("Apply compression")
        out = Dct2.ApplyDct2OnImage(img, size_block, compression_index)
        # Save the image
        println("Save image compressed")
        Utils.SaveBmpImage(out, "output.bmp")

        println("Prepare output")
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
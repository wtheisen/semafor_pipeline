import json
import os
import argparse

def build_html(jsonFile, ServerPort, img_root_dir, numColumns = 1):
    head = "<head>\n\
    <style>\n\
    .container {\n\
      height: 20%;\n\
      border: dashed blue 1px;\n\
    \n\
    }\n\
    .root {\n\
      height: 20%;\n\
      border: dashed red 1px;\n\
    \n\
    }\n\
    \n\
    .container img {\n\
      max-height:100%;\n\
      max-width: 100%;\n\
    }\n\
    .root img {\n\
      max-height:100%;\n\
      max-width: 100%;\n\
    }\n\
    #parent {\n\
      display: flex;\n\
      flex-flow: row;\n\
      width: 100%;\n\
      height: auto; \n\
    }\n\
    </style>\n\
    </head>\n"


    with open(jsonFile, 'r') as fp:
        results = json.load(fp)

    nodes = results['nodes']

    imageList = ""

    ncount = 1
    outputName = os.path.basename(jsonFile).split(".")[0]+".html"

    imageList += "<div id=\"parent\">\n"
    # imageList += "<div class=\"root\">\n\
    #                     <img src=\"0.0.0.0:" + str(ServerPort) + "/" + os.path.join(DatasetName,rootNode) + "\" />\n\
    #                   </div>"

    for node in nodes:
        if ncount % numColumns == 0 and ncount > 0:
            imageList += '</div>\n'
        if ncount%numColumns == 0 and ncount > 0 and ncount < len(nodes)-1:
            imageList += "<div id=\"parent\">\n"

        ncount += 1
        fpath = node["file"].replace(img_root_dir, '')

        imageList += "<div class=\"container\">\n\
                <img src=\"http://0.0.0.0:{}".format(ServerPort) + os.path.join(fpath) + "\" /> Score: {}\n\
                </div>".format(str(node['nodeConfidenceScore']))

    imageList += "</body>"
    return head + imageList


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queryFile")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument('--rootImgDir')
    parser.add_argument('--outputDir')

    args = parser.parse_args()

    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)

    htmlTXT = build_html(args.queryFile, args.port, args.rootImgDir)
    with open(os.path.join(args.outputDir, os.path.basename(args.queryFile) + '.html'), 'w') as f:
        f.write(htmlTXT)

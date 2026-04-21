/**
 * docs/_pdf_runnings.js
 *
 * Header / footer definitions passed to `markdown-pdf` via the
 * `--runnings-path` flag. The footer prints a centred
 * "<page> / <total>" indicator on every page; the header is empty.
 */

exports.header = {
  height: "0cm",
  contents: function () { return ""; },
};

exports.footer = {
  height: "1.2cm",
  contents: function (pageNum, numPages) {
    return (
      '<div style="' +
        'text-align:center;' +
        'font-family:Georgia, \'Times New Roman\', serif;' +
        'font-size:10pt;' +
        'color:#666;' +
        'padding-top:0.4cm;' +
      '">' +
        pageNum + " / " + numPages +
      "</div>"
    );
  },
};

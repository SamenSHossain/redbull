# ============================================================
# Red Bull Conjoint — theme helpers
#  - rb_theme()       : bslib theme to pass into page_*( theme = ... )
#  - rb_head()        : <head> tags (fonts + custom.css)
#  - rb_header()      : branded header bar
#  - rb_hero_stat()   : "Causal estimate: +0.328" hero callout
#  - rb_pill()        : sidebar footer pill ("64 / 64 respondents")
#  - rb_synth()       : Synthesis tab comparison grid
#  - theme_redbull()  : ggplot2 theme + scales
#  - rb_mpl_style()   : matplotlib rcParams (call once at app start)
# ============================================================
suppressPackageStartupMessages({
  library(bslib)
  library(htmltools)
})
# ---- Brand tokens (mirror www/custom.css) -------------------
rb_tokens <- list(
  navy    = "#0B1E3F",
  red     = "#DB0A40",
  red_2   = "#B0072F",
  yellow  = "#FFC906",
  canvas  = "#F7F7F5",
  card    = "#FFFFFF",
  border  = "#E6E6E1",
  fg      = "#0B1E3F",
  muted   = "#5B6678",
  subtle  = "#8A93A2",
  grid    = "#EEEEEE"
)
# ---- bslib theme -------------------------------------------
rb_theme <- function() {
  bs_theme(
    version    = 5,
    bg         = rb_tokens$canvas,
    fg         = rb_tokens$fg,
    primary    = rb_tokens$red,
    secondary  = rb_tokens$navy,
    success    = rb_tokens$navy,
    info       = rb_tokens$navy,
    warning    = rb_tokens$yellow,
    danger     = rb_tokens$red,
    base_font  = font_google("Inter"),
    code_font  = font_google("JetBrains Mono"),
    heading_font = font_google("Inter")
  )
}
# ---- <head> -------------------------------------------------
rb_head <- function() {
  tags$head(
    tags$meta(name = "viewport", content = "width=device-width, initial-scale=1"),
    tags$link(rel = "preconnect", href = "https://fonts.googleapis.com"),
    tags$link(rel = "preconnect", href = "https://fonts.gstatic.com", crossorigin = NA),
    tags$link(
      rel = "stylesheet",
      href = paste0(
        "https://fonts.googleapis.com/css2",
        "?family=Inter:wght@400;500;600;700",
        "&family=JetBrains+Mono:wght@400;500;600",
        "&display=swap"
      )
    ),
    tags$link(rel = "stylesheet", href = "custom.css")
  )
}
# ---- Header bar --------------------------------------------
rb_header <- function(title = "Red Bull Conjoint",
                     subtitle = "Causal Inference Dashboard",
                     eyebrow  = "v1.0 / 64 respondents") {
  div(
    class = "rb-header",
    span(class = "rb-wordmark", title),
    div(class = "rb-divider"),
    span(class = "rb-subtitle", subtitle),
    span(class = "rb-eyebrow", eyebrow)
  )
}
# ---- Hero stat callout -------------------------------------
rb_hero_stat <- function(eyebrow, number, unit = NULL, note = NULL) {
  div(class = "rb-hero-stat",
    div(class = "rb-eyebrow", eyebrow),
    div(
      span(class = "rb-number", number),
      if (!is.null(unit)) span(class = "rb-unit", paste0("  ", unit))
    ),
    if (!is.null(note)) div(class = "rb-note", note)
  )
}
# ---- Sidebar pill ------------------------------------------
rb_pill <- function(...) span(class = "rb-pill", ...)
# ---- Synthesis grid ----------------------------------------
# rows: list of list(claim=, finding=, change=, tone= "keep"|"strat"|"reframe")
rb_synth <- function(rows) {
  cells <- list(
    div(class = "rb-synth-head", "Deck claim"),
    div(class = "rb-synth-head", "Causal finding"),
    div(class = "rb-synth-head", "Recommendation change")
  )
  for (r in rows) {
    tone <- if (is.null(r$tone)) "" else paste0(" rb-", r$tone)
    cells <- c(cells, list(
      div(class = paste0("rb-synth-cell rb-claim", tone), r$claim),
      div(class = "rb-synth-cell", HTML(r$finding)),
      div(class = "rb-synth-cell", HTML(r$change))
    ))
  }
  do.call(div, c(list(class = "rb-synth"), cells))
}
# ---- ggplot2 theme -----------------------------------------
theme_redbull <- function(base_size = 12) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) return(invisible(NULL))
  ggplot2::theme_minimal(base_size = base_size, base_family = "Inter") +
    ggplot2::theme(
      plot.background  = ggplot2::element_rect(fill = "transparent", colour = NA),
      panel.background = ggplot2::element_rect(fill = "transparent", colour = NA),
      panel.grid.major = ggplot2::element_line(colour = rb_tokens$grid, linewidth = 0.4),
      panel.grid.minor = ggplot2::element_blank(),
      axis.line   = ggplot2::element_line(colour = rb_tokens$fg, linewidth = 0.4),
      axis.ticks  = ggplot2::element_line(colour = rb_tokens$fg, linewidth = 0.4),
      axis.text   = ggplot2::element_text(colour = rb_tokens$muted, size = base_size - 2),
      axis.title  = ggplot2::element_text(colour = rb_tokens$fg, size = base_size - 1, face = "bold"),
      plot.title  = ggplot2::element_text(colour = rb_tokens$fg, face = "bold",
                                          size = base_size + 2, margin = ggplot2::margin(b = 8)),
      legend.background = ggplot2::element_rect(fill = "transparent", colour = NA),
      legend.key        = ggplot2::element_rect(fill = "transparent", colour = NA),
      legend.text       = ggplot2::element_text(colour = rb_tokens$fg, size = base_size - 2),
      strip.text  = ggplot2::element_text(colour = rb_tokens$fg, face = "bold")
    )
}
rb_palette <- c(rb_tokens$navy, rb_tokens$red, rb_tokens$yellow,
                "#1F4FA8", "#5B6678", "#8A93A2")
scale_color_redbull <- function(...) ggplot2::scale_color_manual(values = rb_palette, ...)
scale_fill_redbull  <- function(...) ggplot2::scale_fill_manual(values = rb_palette, ...)
# ---- matplotlib rcParams (for reticulate / Python plots) ---
# Returns a list you can apply with:
#   reticulate::py_run_string("import matplotlib as mpl; mpl.rcParams.update(rb)")
rb_mpl_style <- function() {
  list(
    `figure.facecolor` = "none",
    `axes.facecolor`   = "none",
    `savefig.facecolor`= "none",
    `savefig.transparent` = TRUE,
    `font.family`      = "Inter",
    `font.size`        = 11,
    `axes.edgecolor`   = rb_tokens$fg,
    `axes.labelcolor`  = rb_tokens$fg,
    `axes.titlecolor`  = rb_tokens$fg,
    `axes.titleweight` = "bold",
    `axes.spines.top`  = FALSE,
    `axes.spines.right`= FALSE,
    `xtick.color`      = rb_tokens$muted,
    `ytick.color`      = rb_tokens$muted,
    `grid.color`       = rb_tokens$grid,
    `grid.linewidth`   = 0.4,
    `axes.grid`        = TRUE,
    `axes.prop_cycle`  = sprintf("cycler('color', ['%s','%s','%s','%s','%s'])",
                                 rb_tokens$navy, rb_tokens$red, rb_tokens$yellow,
                                 "#1F4FA8", rb_tokens$muted)
  )
}

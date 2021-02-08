library(tidyverse)
library(ggthemes)
library(viridis)
library(latex2exp)

read_csv("data/cordatasr.csv") %>%
    mutate(md12 = map_chr(md12, ~ sprintf("$m^d_{12} = %.1f$", .x)),
           md21 = map_chr(md21, ~ sprintf("$m^d_{21} = %.1f$", .x))) %>%
    ggplot(aes(s1, rhoS,
               color = factor(rho),
               linetype = factor(sigma))) +
    geom_line() +
    scale_color_colorblind() +
    facet_grid(md12 ~ md21,
               labeller = function (labels, multi_line = TRUE) {
                   label_parsed(map(labels, ~ TeX(.x)),
                                multi_line)
               }) +
    scale_y_continuous(breaks = seq(-0.4, 0.8, by = 0.4)) +
    theme_tufte() +
    theme(text = element_text(size = 16),
          panel.grid.major.y = element_line(color = "grey80")) +
    labs(x = TeX("$s_1$"),
         y = TeX("$\\rho_s$"),
         color = TeX("$\\rho_a$"),
         linetype = TeX("$\\sigma$")) +
    coord_cartesian(xlim = c(0, 1))
    
ggsave("/tmp/fig_rhoS_vs_S1.pdf")

logodds <- function (p) {
    ## log( p / (1 - p) )
    log(p) - log1p(-p)
}

read_csv("data/cordatasr.csv") %>%
    mutate(md12 = map_chr(md12, ~ sprintf("$m^d_{12} = %.1f$", .x)),
           md21 = map_chr(md21, ~ sprintf("$m^d_{21} = %.1f$", .x))) %>%
    ggplot(aes(logodds(pd1), rhoS,
               color = factor(rho),
               linetype = factor(sigma))) +
    geom_line() +
    scale_color_colorblind() +
    facet_grid(md12 ~ md21,
               labeller = function (labels, multi_line = TRUE) {
                   label_parsed(map(labels, ~ TeX(.x)),
                                multi_line)
               }) +
    scale_y_continuous(breaks = seq(-0.4, 0.8, by = 0.4)) +
    theme_tufte() +
    theme(text = element_text(size = 16),
          panel.grid.major.y = element_line(color = "grey80")) +
    labs(x = TeX("$\\mathrm{logit}(\\pi_{d\\cdot})$"),
         y = TeX("$\\rho_s$"),
         color = TeX("$\\rho_a$"),
         linetype = TeX("$\\sigma$")) +
    coord_cartesian(xlim = c(-12, 12))

ggsave("/tmp/fig_rhoS_vs_logodds.pdf")

read_csv("data/cordata.csv") %>%
    ggplot(aes(a0, pd1,
               color = factor(rho),
               linetype = factor(sigma))) +
    geom_line() +
    scale_color_colorblind() +
    facet_grid(md12 ~ md21) +
    scale_x_log10() +
    scale_y_continuous(breaks = seq(-0.4, 0.8, by = 0.4)) +
    theme_tufte()


read_csv("data/cordata2.csv") %>%
    ggplot(aes(a1, a2, fill = rhoS)) +
    geom_tile() +
    scale_x_log10() +
    scale_y_log10() +
    facet_grid(md12 ~ md21) +
    scale_fill_viridis() +
    theme_tufte()

suzuki_boundaries <- function(a1, a2, ms12, ms21, md12, md21, d1, d2) {
    b1 <- d1 - md12 * d2
    b2 <- d2 - md21 * d1
    tribble(~b1, ~type, ~b2,
            ifelse(a2 >= b2, b1 - ms12 * (a2 - b2), NA), "ds_ss", a2,
            a1, "ds_dd", ifelse(a1 <= b1, b2 - md21 * (a1 - b1), NA),
            ifelse(a2 <= b2, b1 - md12 * (a2 - b2), NA), "sd_dd", a2,
            a1, "sd_ss", ifelse(a1 >= b1, b2 - ms21 * (a1 - b1), NA))
}

bound_df <-
    read_csv("data/cordata2.csv") %>%
    distinct(a1, a2, md12, md21, sigma, rho) %>%
    mutate(boundary = pmap(list(a1, a2, md12, md21),
                           function(a1, a2, md12, md21)
                               suzuki_boundaries(a1, a2, 0, 0, md12, md21, 1, 1))) %>%
    unnest(boundary) %>%
    mutate(md12 = map_chr(md12, ~ sprintf("$m^d_{12} = %.1f$", .x)),
           md21 = map_chr(md21, ~ sprintf("$m^d_{21} = %.1f$", .x)))

read_csv("data/cordata2.csv") %>%
    ## mutate(d1 = 1, d2 = 1) %>%
    ## filter(a1 >= d1 - md12 * d2) %>%
    ## filter(a2 >= d2 - md21 * d1) %>%
    mutate(md12 = map_chr(md12, ~ sprintf("$m^d_{12} = %.1f$", .x)),
           md21 = map_chr(md21, ~ sprintf("$m^d_{21} = %.1f$", .x))) %>%
    ggplot() +
    geom_raster(aes(a1, a2, fill = rhoS)) +
    scale_x_log10() +
    scale_y_log10() +
    facet_grid(md12 ~ md21,
               labeller = function (labels, multi_line = TRUE) {
                   label_parsed(map(labels, ~ TeX(.x)),
                                multi_line)
               }) +
    scale_fill_viridis(option = "magma") +
    geom_abline(intercept = 0, slope = 1,
                alpha = 0.6,
                color = "darkred") +
    geom_line(aes(b1, b2,
                  group = type),
              color = "lightgrey",
              alpha = 0.6,
              data = bound_df) +
    theme_tufte() +
    theme(text = element_text(size = 16)) +
    labs(x = TeX("$a_1$"),
         y = TeX("$a_2$"),
         fill = TeX("$\\rho_S$")) +
    coord_cartesian(xlim = c(0.15, 2.5),
                    ylim = c(0.15, 2.5))

ggsave("/tmp/fig_rhoS_vs_a1_a2.pdf")

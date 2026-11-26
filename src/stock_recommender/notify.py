"""Email the recommendations through the SMTP server in the settings."""
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from html import escape
import logging
import smtplib
import ssl
from .config import Settings
from .schemas import EmailReport, Recommendation, RecommendationList

log = logging.getLogger("uvicorn.error")  # Uvicorn's general logger, which its console handler prints.
HEADERS = ["#", "Ticker", "News", "Close"]


def cells(item: Recommendation) -> list[str]:
    return [str(item.rank), item.ticker, str(item.news_count), f"{item.close:,.2f}"]


def exit_text(body: RecommendationList) -> str:
    """When to sell what the lists buy: each model's exit strategy in words, named when there are several."""
    if not body.exits:
        return f"hold {body.horizon_days} trading days"
    if len(body.exits) == 1:
        return next(iter(body.exits.values())).summary
    return "; ".join(f"{name}: {strategy.summary}" for name, strategy in body.exits.items())


def compose(body: RecommendationList, sender: str, recipients: list[str]) -> EmailMessage:
    """Return a message listing the buys and sells, as plain text and as HTML tables."""
    # A served backtest run is named, and with several models each row says whose it is.
    source = f" from backtest run {body.run}" if body.run else ""
    several = len({item.model for item in body.buy + body.sell}) > 1
    headers = HEADERS[:1] + ["Model"] + HEADERS[1:] if several else HEADERS
    intro = (f"Stocks to buy and sell at today's close (16:00 New York), from bars through {body.as_of} and news published "
             f"before {body.news_cutoff:%Y-%m-%d %H:%M} UTC. Model {body.model_version}{source}. "
             f"Exit: {exit_text(body)}.")
    text, html = [intro], [f"<p>{escape(intro)}</p>"]
    if body.fetch and body.fetch.errors:  # The lists come from the older data still on disk.
        warning = "Warning: fetching new data failed, so these lists may be stale: " + "; ".join(body.fetch.errors)
        text.append(warning)
        html.append(f"<p><b>{escape(warning)}</b></p>")
    for title, items in (("Buy", body.buy), ("Sell", body.sell)):
        rows = [cells(item)[:1] + [item.model or ""] + cells(item)[1:] if several else cells(item) for item in items]
        widths = [max(map(len, column)) for column in zip(headers, *rows)]
        text += ["", title.upper()] + ["  ".join(cell.rjust(width) for cell, width in zip(line, widths))
                                       for line in [headers, *rows]]
        header = "".join(f"<th>{escape(cell)}</th>" for cell in headers)
        lines = "".join("<tr>" + "".join(f'<td align="right">{escape(cell)}</td>' for cell in row) + "</tr>"
                        for row in rows)
        html.append(f'<h3>{title}</h3><table cellpadding="4"><tr>{header}</tr>{lines}</table>')
    message = EmailMessage()
    message["Subject"] = f"Stock recommendations as of {body.as_of}: {len(body.buy)} buy, {len(body.sell)} sell"
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid()
    message.set_content("\n".join(text))
    message.add_alternative(f"<html><body>{''.join(html)}</body></html>", subtype="html")
    return message


def send(body: RecommendationList, settings: Settings, recipients: list[str]) -> EmailReport:
    """Email the recommendations to the recipients; a failure is reported, not raised."""
    sender = settings.email_from or settings.smtp_user
    if not (settings.smtp_host and sender):
        return EmailReport(sent_to=[], error="Email is not configured: set RECOMMENDER_SMTP_HOST and "
                                             "RECOMMENDER_SMTP_USER or _EMAIL_FROM.")
    if not recipients:
        return EmailReport(sent_to=[], error="No address to send the lists to.")
    context = ssl.create_default_context()  # Verifies the server's certificate.
    try:
        message = compose(body, sender, recipients)
        if settings.smtp_port == 465:
            server = smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=30, context=context)
        else:
            server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30)
        with server:
            server.ehlo()
            encrypted = settings.smtp_port == 465
            if not encrypted and server.has_extn("starttls"):
                server.starttls(context=context)
                server.ehlo()
                encrypted = True
            if settings.smtp_user:
                if not encrypted:
                    raise smtplib.SMTPException("The server offers no TLS; refusing to send the password unencrypted.")
                server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(message)
    except (OSError, smtplib.SMTPException) as error:
        log.error("Email failed: %s", error)
        return EmailReport(sent_to=[], error=f"{type(error).__name__}: {error}")
    except ValueError as error:  # Non-ASCII credentials or host, or a line break in an address.
        log.error("Email failed: invalid email settings (%s)", type(error).__name__)
        # The error's text can quote the password, so it is left out.
        return EmailReport(sent_to=[], error=f"{type(error).__name__}: invalid email settings; the SMTP host, user, "
                                             "password and addresses must be ASCII without line breaks.")
    return EmailReport(sent_to=recipients)
